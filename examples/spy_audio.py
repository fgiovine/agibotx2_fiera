import rclpy, threading, time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from aimdk_msgs.msg import AudioPlayback
from aimdk_msgs.srv import PlayTts
from aimdk_msgs.msg import PlayTtsRequest, CommonRequest

rclpy.init()
node = rclpy.create_node('spy')
qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10, durability=DurabilityPolicy.VOLATILE)

def cb(msg):
    print(f'ch={msg.info.channels} rate={msg.info.sample_rate} size={msg.info.size} fmt={msg.info.sample_format} coding={msg.info.coding_format} pkg={msg.pkg_name} data={len(msg.data.data)}')

node.create_subscription(AudioPlayback, '/aima/hal/audio/playback', cb, qos)
client = node.create_client(PlayTts, '/aimdk_5Fmsgs/srv/PlayTts')
client.wait_for_service(5.0)

def trigger():
    time.sleep(2.0)
    req = PlayTts.Request()
    req.header = CommonRequest()
    req.tts_req = PlayTtsRequest()
    req.tts_req.text = 'Ciao, sono Ciruzzo, piacere di conoscerti!'
    req.tts_req.trace_id = 'x'
    req.tts_req.domain = 'it'
    client.call_async(req)
    print('TTS triggered')

threading.Thread(target=trigger, daemon=True).start()
print('Aspetto audio per 10s...')
end = time.time() + 10
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.1)
rclpy.shutdown()
