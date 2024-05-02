#! /usr/bin/env python
#  The code was adapted from https://www.theconstructsim.com/ros-projects-exploring-ros-using-2-wheeled-robot
#  Or from lecture handouts

# import ros stuff
import rospy
import numpy as np
# import ros message
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf import transformations
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
# import ros service
from std_srvs.srv import SetBool
from com760_group19.srv import Com760Group19Status
from com760_group19.msg import Com760Group19Custom

import math

class Bugs():
    def __init__(self):
        rospy.init_node('bugs')
        rospy.wait_for_service('/go_to_point_switch')
        rospy.wait_for_service('/wall_follower_switch')
        rospy.wait_for_service('/gazebo/set_model_state')
        self.starting_point = Point()
        self.closest_point = Point()
        self.go_to_goal = rospy.ServiceProxy('/go_to_point_switch', Com760Group19Status)
        self.follow_wall = rospy.ServiceProxy('/wall_follower_switch', Com760Group19Status)
        self.homing_signal = rospy.Publisher("homing_signal", Com760Group19Custom, queue_size=1)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.sub_scan = rospy.Subscriber('/group19Bot/laser/scan', LaserScan, self.callback_laser)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.callback_odom)
        self.algorithm = rospy.get_param('algo')
        self.yaw = 0
        self.error = 5 * (math.pi / 180)
        self.position = Point()
        self.initial = Point()
        self.initial.x = rospy.get_param('initial_x')
        self.initial.y = rospy.get_param('initial_y')
        self.initial.z = rospy.get_param('initial_z')
        self.goal = Point()
        self.goal.x = rospy.get_param('des_pos_x')
        self.goal.y = rospy.get_param('des_pos_y')
        self.goal.z = 0
        self.goal_yaw = 0
        self.goal_reached = False
        self.home_reached = False
        self.scan = None
        self.description = ['Go to goal', 'following wall']
        self.state = 0
        self.state_time_count = 0 # seconds the group19Bot is in a state
        self.loop_count = 0
        self.rate = rospy.Rate(20)
        self.max_linear_velocity = 0.7  # Maximum linear velocity
        self.max_angular_velocity = 0.4  # Maximum angular velocity
        self.min_distance = 0.5  # Minimum distance to obstacles for full speed
        self.max_distance = 2.0  # Maximum distance to obstacles for no speed
        self.log = rospy.get_param('log')
        self.target_count = 0

    # callbacks
    def callback_odom(self, msg):
        # position
        self.position = msg.pose.pose.position

        # yaw
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        euler = transformations.euler_from_quaternion(quaternion)
        self.yaw = euler[2]

    def callback_laser(self, msg):
        self.scan = {
            'right':  min(min(msg.ranges[0:143]), 10),
            'front_right': min(min(msg.ranges[144:287]), 10),
            'front':  min(min(msg.ranges[288:431]), 10),
            'front_left':  min(min(msg.ranges[432:575]), 10),
            'left':   min(min(msg.ranges[576:719]), 10),
        }

    def change_behaviour(self, state):
        self.state_time_count = 0
        self.state = state
        if self.state == 0:
            resp = self.go_to_goal(True, self.set_linear_velocity(), self.set_angular_velocity())
            resp = self.follow_wall(False, self.set_linear_velocity(), self.set_angular_velocity())
        if self.state == 1:
            resp = self.go_to_goal(False, self.set_linear_velocity(), self.set_angular_velocity())
            resp = self.follow_wall(True, self.set_linear_velocity(), self.set_angular_velocity())
        if self.algorithm == 'bug1':
            self.description = ['Go to goal', 'Go around obsactle', 'Go to goal']
            if self.state == 2:
                resp = self.go_to_goal(True, self.set_linear_velocity(), self.set_angular_velocity())
                resp = self.follow_wall(False, self.set_linear_velocity(), self.set_angular_velocity())
        log = "behaviour changed to: %s" % self.description[state]
        rospy.loginfo(log)
        rospy.loginfo(resp)
        return resp

    def isclose(self, x, y, rel_tol):
        if abs(x-y) <= rel_tol:
            return True
        else:
            return False
    
    def target_reached(self):
        if self.isclose(self.goal.x, self.position.x, rel_tol=0.3) \
                and self.isclose(self.goal.y, self.position.y, rel_tol=0.3) \
                    and self.state_time_count > 5 and self.target_count == 0:
            self.goal_reached = True
            rospy.loginfo('Goal Reached')
            self.target_count +=1
            self.reset_positions()
            rospy.loginfo('Position and goal reset')
            return True
        elif self.isclose(self.goal.x, self.position.x, rel_tol=0.3) \
                and self.isclose(self.goal.y, self.position.y, rel_tol=0.3) \
                    and self.state_time_count > 5 and self.target_count == 1:
            self.home_reached = True
            rospy.loginfo('Looks like we made it')
            self.target_count +=1
            rospy.loginfo('Position and goal reset')
            return True
        else:
            return False
    
    def send_homing_signal(self, state, message='Come home'):
        # Blocks until registered with master
        msg = Com760Group19Custom()
        msg.state = state
        msg.goal_x = self.initial.x   # i.e. initial.x   -> goal.x
        msg.goal_y = self.initial.y # i.e. initial.y   -> goal.y
        msg.message = message
        self.homing_signal.publish(msg)
        rospy.loginfo('Homing message sent')
    
    def reset_positions(self):
        x = self.initial.x 
        y = self.initial.y
        gx = self.goal.x
        gy = self.goal.y
        self.initial.x = gx
        self.initial.y = gy
        self.goal.x = x
        self.goal.y = y
    
    def proportional_controller(self):
        # Porportional Controller
        # Linear velocity in the x-axis:
        cmd_vel = Twist()
        cmd_vel.linear.x = 1.5 * math.sqrt(pow((self.goal.x - self.position.x), 2) + pow((self.goal.y - self.position.y), 2))
        cmd_vel.linear.y = 0
        cmd_vel.linear.z = 0

        # Angular velocity in the z-axis:
        cmd_vel.angular.x = 0
        cmd_vel.angular.y = 0
        cmd_vel.angular.z = 6 * (math.atan2(self.goal.y - self.position.y, self.goal.x - self.position.x) - self.position.z) # z is theta? right?

        # Log the velocity
        if self.log == 'true':
            log = {'Proportional controller cmd_vel': cmd_vel}
            rospy.loginfo(log)
        return cmd_vel

    def distance_between_points(self):
        # p0 is the current position
        # p1 and p2 points define the line
        p0 = self.position
        p1 = self.initial
        p2 = self.goal
        p0 = np.array([p0.x, p0.y])
        p1 = np.array([p1.x, p1.y])
        p2 = np.array([p2.x, p2.y])
        
        # Calculate the numerator of the distance formula
        numerator = np.abs((p2[1] - p1[1]) * p0[0] - (p2[0] - p1[0]) * p0[1] \
                           + (p2[0] * p1[1]) - (p2[1] * p1[0]))
        
        # Calculate the euclidean distance
        denominator = np.sqrt(np.sum(np.power(p2 - p1, 2)))
        
        # Calculate the distance
        distance = numerator / denominator
        
        return distance
    
    def euclidean_distance(self, p1, p2):
        dist = math.sqrt((p1.y - p2.y)**2 + (p1.x - p2.x)**2)
        return dist

    def normalize_angle(self, angle):
        if(math.fabs(angle) > math.pi):
            angle = angle - (2 * math.pi * angle) / (math.fabs(angle))
        return angle
    
    def calculate_goal_yaw(self):
        # Calculate the direction towards the goal
        x = self.goal.x - self.position.x
        y = self.goal.y - self.position.y
        self.goal_yaw = math.atan2(y, x)
    
    def reset_model(self):
        # set group19Bot position
        model_state = ModelState()
        model_state.model_name = 'group19Bot'
        model_state.reference_frame = 'world'
        model_state.pose.position.x = self.initial.x
        model_state.pose.position.y = self.initial.y
        # model_state.pose.position.z = self.initial.z
        resp = self.set_model_state(model_state)
        log = "Model reset to: %s" %model_state
        rospy.loginfo(log)
        rospy.loginfo(resp)
    
    def bug_zero(self):
        # set group19Bot position
        self.reset_model()
        # initialize going to goal
        while self.scan == None:
                continue
        self.change_behaviour(0)
        
        while not rospy.is_shutdown():
            
            if self.state == 0: # Go to goal
                if self.scan['front'] > 0.15 and self.scan['front'] < 1:
                    self.change_behaviour(1) # Following wall
            
            elif self.state == 1: # Following wall
                desired_yaw = math.atan2(self.goal.y - self.position.y, self.goal.x - self.position.x)
                err_yaw = self.normalize_angle(desired_yaw - self.yaw)
                
                # less than 30 degrees
                if math.fabs(err_yaw) < (math.pi / 6) and \
                    self.scan['front'] > 1.5 and self.scan['front_right'] > 1 and self.scan['front_left'] > 1:
                    rospy.loginfo('less than 30')
                    self.change_behaviour(0) # Go to goal
                
                # between 30 and 90
                if err_yaw > 0 and \
                    math.fabs(err_yaw) > (math.pi / 6) and \
                    math.fabs(err_yaw) < (math.pi / 2) and \
                    self.scan['left'] > 1.5 and self.scan['front_left'] > 1:
                    rospy.loginfo('between 30 and 90 - to the left')
                    self.change_behaviour(0) # Go to goal
                    
                if err_yaw < 0 and \
                    math.fabs(err_yaw) > (math.pi / 6) and \
                    math.fabs(err_yaw) < (math.pi / 2) and \
                    self.scan['right'] > 1.5 and self.scan['front_right'] > 1:
                    rospy.loginfo('between 30 and 90 - to the right')
                    self.change_behaviour(0) # Go to goal
                
            self.rate.sleep()

    def bug_one(self):
        # set group19Bot position
        self.reset_model()
        # initialize going to goal
        while self.scan == None:
                continue
        self.change_behaviour(0) # Go to goal
        
        while not rospy.is_shutdown():
            
            if self.state == 0: # Go to goal
                if self.scan['front'] > 0.15 and self.scan['front'] < 1:
                    self.closest_point = self.position
                    self.starting_point = self.position
                    rospy.loginfo('Closest point updated to: %s' %self.closest_point)
                    self.change_behaviour(1) # Go arround obsactle
            
            elif self.state == 1: # Go arround obsactle
                # If current position is closer to the goal than the previous
                # closest_position, assign current position to closest_point
                if self.euclidean_distance(self.position, self.goal) \
                    < self.euclidean_distance(self.closest_point, self.goal):
                    self.closest_point = self.position
                    if self.log == True:
                        rospy.loginfo('Closest point updated to: %s' %self.closest_point)
                    
                # compare only after 5 seconds - need some time to get out of starting_point
                # If group19Bot reaches (is close to) starting point
                if self.state_time_count > 5 and \
                self.euclidean_distance(self.position, self.starting_point) < 0.2:
                    self.change_behaviour(2) # Go to goal
            
            elif self.state == 2: # Go to goal
                # If group19Bot reaches the closest_point
                if self.euclidean_distance(self.position, self.closest_point) < 0.2:
                    self.change_behaviour(0) # Go to goal
                    
            self.loop_count += 1
            if self.loop_count == 20:
                self.state_time_count += 1
                self.loop_count = 0
                
            self.rate.sleep()

    def bug_two(self):
        # set group19Bot position
        self.reset_model()

        # initialize going to goal
        while self.scan == None:
                continue
        resp = self.change_behaviour(0) # Go to goal
        # TODO 'check this is needed'
        # rospy.sleep(5)
        while not rospy.is_shutdown() and not self.home_reached:

            distance_position_to_line = self.distance_between_points()

            if self.state == 0: # Go to goal
                if self.scan['front'] > 0.15 and self.scan['front'] < 1:
                    resp = self.change_behaviour(1) # Following wall
                elif self.target_reached():
                    rospy.loginfo('Sending the homing signal from state 0')
                    self.send_homing_signal(3, 'Come home')
                    resp = self.change_behaviour(0) # Go to goal

                    
            elif self.state == 1: # Following wall
                if self.state_time_count > 5 and distance_position_to_line <= 0.3:
                    rospy.loginfo("Switching to tangent bug mode")
                    self.calculate_goal_yaw()
                    if self.log == 'true':
                        rospy.loginfo("Goal direction: [%.2f]", self.goal_yaw)
                    resp = self.change_behaviour(0) # Go to goal
                    if self.target_reached():
                        rospy.loginfo('Sending the homing signal from state 1')
                        self.send_homing_signal(3, 'Come home')
                        resp = self.change_behaviour(0) # Go to goal
            
            self.loop_count = self.loop_count + 1
            if self.loop_count == 20:
                self.state_time_count = self.state_time_count + 1
                self.loop_count = 0
            
            if self.log == True:
                rospy.loginfo("distance to line: [%.2f], position: [%.2f, %.2f]", \
                            self.distance_between_points(), self.position.x, self.position.y)
                rospy.loginfo("Current state of bug behaviour is: %s" %self.description[self.state])
            self.rate.sleep()

    def run(self):
        if self.algorithm == 'bug0':
            self.bug_zero()
        elif self.algorithm == 'bug1':
            self.bug_one()
        elif self.algorithm == 'bug2':
            self.bug_two()
        else:
            rospy.loginfo('Set the algorithm param in the bugs launch file or launch command')

if __name__ == '__main__':
    try:
        bugs= Bugs()
        bugs.run()
    except rospy.ROSInterruptException:
        pass

