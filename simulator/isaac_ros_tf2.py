import rospy
import tf2_ros
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3


class Send_data:
    def __init__(self):
        #rospy.init_node('tf_listener')
        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub = rospy.Publisher('lscm/odom', Odometry, queue_size=1)
        
    def send_odom(self, lv, av, angle):
        try:
            #self.tfBuffer.waitForTransform('world', 'agv_base_link', rospy.Time.now(), rospy.Duration(0.5))
            trans = self.tfBuffer.lookup_transform('world', 'agv_base_link', rospy.Time.now())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("in exception!!!")
            return False
        print("trans is ", trans)
        current_time = rospy.Time.now()
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.pose.pose = Pose(Point(trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z), Quaternion(trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w))
        odom.child_frame_id = "base_link"
        
        odom.twist.twist = Twist(Vector3(lv*math.cos(angle),-lv*math.sin(angle),0), Vector3(0,0,av))
        #odom.twist.twist = Twist(lv, av)
        #print("lv is ", lv)
        #print("av is ", av)
        print("odom twist is ", odom.twist.twist)
        self.pub.publish(odom)
        return True

