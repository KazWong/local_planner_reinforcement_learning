import rospy
import tf
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3


class Send_data:
    def __init__(self):
        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher('lscm/odom', Odometry, queue_size=1)
        
    def init_transform(self):
        #self.listener.waitForTransform("world", "agv_base_link", rospy.Time(0), rospy.Duration(0.5))
        #self.listener.canTransform("world", "agv_base_link", rospy.Time.now())
        try:
            (trans,rot) = self.listener.lookupTransform('world', 'agv_base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("in exception in init!!!")
        print("can transform???")
        
    def send_odom(self, lv, av, angle):
        #print("send odom!!!!!!")
        #self.listener.waitForTransform("world", "agv_base_link", rospy.Time.now(), rospy.Duration(0.5))
        try:
            #(trans,rot) = self.listener.lookupTransform('world', 'agv_base_link', rospy.Time.now())
            (trans,rot) = self.listener.lookupTransform('world', 'agv_base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("in exception!!!")
            return False
        #print("trans is ", trans)
        #print("rot is ", rot)
        
        current_time = rospy.Time.now()
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.pose.pose = Pose(Point(trans[0], trans[1], trans[2]), Quaternion(rot[0], rot[1], rot[2], rot[3]))
        odom.child_frame_id = "base_link"
        #odom.twist.twist = Twist(Vector3(lv*math.cos(angle),-lv*math.sin(angle),0), Vector3(0,0,av))
        odom.twist.twist = Twist(Vector3(lv,0,0), Vector3(0,0,av))
        print("odom twist is ", odom.twist.twist)
        self.pub.publish(odom)
        return True

