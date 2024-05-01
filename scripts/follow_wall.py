#! /usr/bin/env python
#  The code was adapted from https://www.theconstructsim.com/ros-projects-exploring-ros-using-2-wheeled-robot
#  Or from lecture handouts

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf import transformations
from std_srvs.srv import *
from com760_group19.srv import Com760Group19Status, Com760Group19StatusResponse

import math

class WallFollower():
    def __init__(self):
        rospy.init_node('Wall_follower')
        self.srv = rospy.Service('wall_follower_switch', Com760Group19Status, self.wall_follower_switch)
        self.scan_sub = rospy.Subscriber('/group19Bot/laser/scan', LaserScan, self.scan_callback)
        self.cmd_pub = rospy.Publisher('/group19Bot/cmd_vel', Twist, queue_size=1)
        self.active = False
        self.state = 0
        self.state_dict = {
            0: 'find the wall',
            1: 'turn left',
            2: 'follow the wall',
        }
        self.res = Com760Group19StatusResponse()
        self.scan = []
        self.rate = rospy.Rate(20)

    def wall_follower_switch(self, req):
        self.active = req.flag
        self.res.success = True
        self.res.message = 'Done!'
        return self.res

    def scan_callback(self, cmd_vel):
        self.scan = [
            min(min(cmd_vel.ranges[0:143]), 10),
            min(min(cmd_vel.ranges[144:287]), 10),
            min(min(cmd_vel.ranges[288:431]), 10),
            min(min(cmd_vel.ranges[432:575]), 10),
            min(min(cmd_vel.ranges[576:719]), 10),
        ]
        scan = {
            'right':  min(min(cmd_vel.ranges[0:143]), 10),
            'front and right': min(min(cmd_vel.ranges[144:287]), 10),
            'front':  min(min(cmd_vel.ranges[288:431]), 10),
            'front and left':  min(min(cmd_vel.ranges[432:575]), 10),
            'left':   min(min(cmd_vel.ranges[576:719]), 10),
        }
        self.take_action(scan)

    def update_state(self, state):
        if state is not self.state:
            rospy.loginfo('Wall follower state updated to - [%s] - %s' % (state, self.state_dict[state]))
            self.state = state

    def take_action(self, scan):
        cmd_vel = Twist()
        linear_x = 0
        angular_z = 0
        
        description = ''

        d = 1.5
        
        if scan['front'] > d and scan['front and left'] > d and scan['front and right'] > d:
            description = 'case 1 - nothing'
            linear_x = 0.6 # Go straight ahead
            angular_z = 0 # No turn
            self.update_state(0) # Find the wall
        elif scan['front'] < d and scan['front and left'] > d and scan['front and right'] > d:
            description = 'case 2 - front'
            linear_x = 0 # Stop
            angular_z = 0.3 # Turn left
            self.update_state(1) # Turn
        elif scan['front'] > d and scan['front and left'] > d and scan['front and right'] < d:
            description = 'case 3 - Obstacle is front and right'
            linear_x = 0 # Stop
            angular_z = 0.3 # Turn left
            self.update_state(2)
        elif scan['front'] > d and scan['front and left'] < d and scan['front and right'] > d:
            description = 'case 4 - Obstacle is front and left'
            linear_x = 0 # Stop
            angular_z = -0.3  # Turn right
            self.update_state(0) # Find the wall
        elif scan['front'] < d and scan['front and left'] > d and scan['front and right'] < d:
            description = 'case 5 - Obstacle is front and front and right'
            linear_x = 0 # Stop
            angular_z = 0.3 # Turn left
            self.update_state(1) # Stop and Turn
        elif scan['front'] < d and scan['front and left'] < d and scan['front and right'] > d:
            description = 'case 6 - Obstacle is front and front and left'
            linear_x = 0 # Stop
            angular_z = -0.3 # Turn right
            self.update_state(1) # Stop and Turn
        elif scan['front'] < d and scan['front and left'] < d and scan['front and right'] < d:
            description = 'case 7 - Obstacle is front and front and left and front and right'
            linear_x = 0 # Stop
            angular_z = 0.3 # Turn left
            self.update_state(1) # Turn
        elif scan['front'] > d and scan['front and left'] < d and scan['front and right'] < d:
            description = 'case 8 - Obstacle is front and left and front and right'
            linear_x = 0.3 # Go straight ahead
            angular_z = 0 # No turn
            self.update_state(0) # Find the wall
        else:
            description = 'unknown case'
        
        rospy.loginfo(scan)
        rospy.loginfo(description)

        cmd_vel.linear.x = linear_x
        cmd_vel.angular.z = angular_z
        # self.cmd_pub.publish(cmd_vel)
        # rospy.loginfo('Published cmd_vel: [%s]' % cmd_vel)


    def min_avoidance(self):
        cmd_vel = Twist()
        # Back up if the scan is bad, or if we are too close. 
        if(math.isnan(self.range_ahead) or self.range_ahead < self.target):
            cmd_vel.linear.x = -1.0
            cmd_vel.angular.z = 1.0   
        else:
            cmd_vel.linear.x = 1
            cmd_vel.angular.z = 0   
        
        rospy.loginfo("Range ahead is : {:.3f}".format(self.range_ahead))
        self.cmd_pub.publish(cmd_vel)
        rospy.loginfo('Published cmd_vel: [%s]' % self.cmd_vel)

    # State 0
    def find_wall(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2
        cmd_vel.angular.z = -0.3
        return cmd_vel

    # State 1
    def turn(self):
        cmd_vel = Twist()
        cmd_vel.angular.z = 0.3 # Turn left
        return cmd_vel

    # State 2
    def follow_the_wall(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 1
        return cmd_vel

    def run(self):
        while not rospy.is_shutdown():
            if not self.active:
                self.rate.sleep()
                continue
                
            cmd_vel = Twist()
            if self.state == 0:
                cmd_vel = self.find_wall()
            elif self.state == 1:
                cmd_vel = self.turn()
            elif self.state == 2:
                cmd_vel = self.follow_the_wall()
                pass
            else:
                rospy.logerr('Unknown state!')
            self.cmd_pub.publish(cmd_vel)
            rospy.loginfo('Published cmd_vel: [%s]' % cmd_vel)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        wall_follower = WallFollower()
        wall_follower.run()
    except rospy.ROSInterruptException:
        pass