#! /usr/bin/env python
#  The code was adapted from https://www.theconstructsim.com/ros-projects-exploring-ros-using-2-wheeled-robot
#  Or from lecture handouts

# import ros stuff
import rospy
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf import transformations
from com760_group19.srv import Com760Group19Status, Com760Group19StatusResponse
from com760_group19.msg import Com760Group19Custom


import math

class GoToPoint():
    def __init__(self):
        rospy.init_node('go_to_point')
        self.srv = rospy.Service('go_to_point_switch', Com760Group19Status, self.go_to_point_switch)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.callback_odom)
        self.cmd_pub = rospy.Publisher('/group19Bot/cmd_vel', Twist, queue_size=1)
        self.homing_signal = rospy.Subscriber('/homing_signal', Com760Group19Custom, self.homing_callback)
        self.active = False
        # robot state variables
        self.position = Point()
        self.yaw = 0
        # robot state
        self.state = 0
        self.initial = Point()
        # goal
        self.goal = Point()
        self.goal.x = rospy.get_param('des_pos_x')
        self.goal.y = rospy.get_param('des_pos_y')
        self.goal.z = 0
        self.goal_reached = False
        # parameters
        self.yaw_precision = math.pi / 90 # +/- 2 degree allowed
        self.dist_precision = 0.3
        self.start_time = rospy.Time.now()
        self.timeout = rospy.Duration(10)
        self.rate = rospy.Rate(20)

    # homing signal callback
    def homing_callback(self, msg):
        rospy.loginfo(msg)
        self.state = msg.state
        self.goal.x = msg.goal_x
        self.goal.y = msg.goal_y

    # service callbacks
    def go_to_point_switch(self, req):
        rospy.loginfo(req)
        self.active = req.flag
        res = Com760Group19StatusResponse()
        res.success = True
        res.message = 'Done!'
        return res

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
        rospy.loginfo('Orientaton for position.z is [%s]' % self.yaw)

    def update_state(self, state):
        self.state = state
        rospy.loginfo('State changed to [%s]' % self.state)

    def normalize_angle(self, angle):
        if(math.fabs(angle) > math.pi):
            angle = angle - (2 * math.pi * angle) / (math.fabs(angle))
        return angle

    def fix_heading(self, des_pos, x_vel=False):
        desired_yaw = math.atan2(des_pos.y - self.position.y, des_pos.x - self.position.x)
        err_yaw = self.normalize_angle(desired_yaw - self.yaw)
        
        rospy.loginfo('Desired yaw is [%s]' % desired_yaw)
        rospy.loginfo('Error yaw is [%s]' % err_yaw)
        
        cmd_vel = Twist()

        if math.fabs(err_yaw) > self.yaw_precision:
            cmd_vel.angular.z = 0.7 if err_yaw > 0 else -0.7
            
        self.cmd_pub.publish(cmd_vel)
        rospy.loginfo('Published cmd_vel: [%s]' % cmd_vel)
        
        # When yaw error is less than yaw precision
        if math.fabs(err_yaw) <= self.yaw_precision:
            rospy.loginfo('Yaw error: [%s]' % err_yaw)
            self.update_state(1) # Go strtaight ahead

    def go_straight(self, des_pos):
        desired_yaw = math.atan2(des_pos.y - self.position.y, des_pos.x - self.position.x)
        err_yaw = desired_yaw - self.yaw
        err_pos = math.sqrt(pow(des_pos.y - self.position.y, 2) + pow(des_pos.x - self.position.x, 2))
        
        if err_pos > self.dist_precision:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.6
            cmd_vel.angular.z = 0.2 if err_yaw > 0 else -0.2
            self.cmd_pub.publish(cmd_vel)
            rospy.loginfo('Published cmd_vel: [%s]' % cmd_vel)
        else:
            rospy.loginfo('Position error: [%s]' % err_pos)
            self.update_state(2)
        
        # state change conditions
        if math.fabs(err_yaw) > self.yaw_precision:
            rospy.loginfo('Yaw error: [%s]' % err_yaw)
            self.update_state(0)

    def done(self):
        if self.goal_reached:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0
            cmd_vel.angular.z = 0
            self.cmd_pub.publish(cmd_vel)
            rospy.loginfo('Published cmd_vel: [%s]' % cmd_vel)
        else:
            rospy.loginfo('First goal reached')
            self.goal_reached = True
    def run(self):
        while not rospy.is_shutdown():
            if not self.active:
                continue
            else:
                if self.state == 0: # Fix heading
                    self.fix_heading(self.goal)
                elif self.state == 1: # Go straight
                    self.go_straight(self.goal)
                elif self.state == 2: # Goal reached
                    rospy.loginfo('HURRAH, WE DID IT YAY!')
                    self.done()
                elif self.state == 3: # Go home
                    rospy.loginfo('Making the journey home')
                    self.fix_heading(self.goal)
                else:
                    rospy.logerr('Unknown state!')
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        goto = GoToPoint()
        goto.run()
    except rospy.ROSInterruptException:
        pass
