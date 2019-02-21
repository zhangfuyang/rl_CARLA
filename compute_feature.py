import numpy as np
import tensorflow as tf

from environment.carla.client import make_carla_client
from environment.carla.settings import CarlaSettings
from environment.carla.sensor import Camera
from environment.carla.carla_server_pb2 import Control
from environment.carla.tcp import TCPConnectionError
from environment.imitation.imitation_learning import ImitationLearning
import time
import random

sess = tf.Session()
Image_agent = ImitationLearning(sess)
Vec_ = []
Image_agent.load_model()
global_episode = 0
while True:
    try:
        with make_carla_client('localhost', 2000) as client:
            for episode in range(3):
                if episode < global_episode:
                    continue
                print(episode)
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=20,
                    NumberOfPedestrians=40,
                    WeatherId=random.choice([1, 3, 7, 8, 14]),
                    QualityLevel='Epic'
                )
                settings.randomize_seeds()
                camera = Camera('CameraRGB')
                camera.set(FOV=100)
                camera.set_image_size(800, 600)
                camera.set_position(2.0, 0.0, 1.4)
                camera.set_rotation(-15.0, 0, 0)
                settings.add_sensor(camera)

                scene = client.load_settings(settings)
                number_of_player_starts = len(scene.player_start_spots)
                player_start = random.randint(0, max(0, number_of_player_starts - 1))
                client.start_episode(player_start)
                for i in range(300):
                    measurements, sensor_data = client.read_data()
                    if random.randint(0,1) == 0:
                        control = measurements.player_measurements.autopilot_control
                        control.steer += random.uniform(-0.1, 0.1)
                        client.send_control(control)
                    else:
                        client.send_control(
                            steer=random.randint(-1.0, 1.0),
                            throttle=0.5,
                            brake=0.0,
                            hand_brake=False,
                            reverse=False
                        )
                    feature_vec = Image_agent.compute_feature(sensor_data)
                    Vec_.append(feature_vec)
                    if measurements.player_measurements.collision_other != 0 or \
                            measurements.player_measurements.collision_pedestrians != 0 or \
                            measurements.player_measurements.collision_vehicles != 0:
                        break
                global_episode += 1

        break
    except TCPConnectionError as error:
        print(error)
        time.sleep(5.0)
V = np.array(Vec_)
print('max', V.max(axis=0))
print('min', V.min(axis=0))
print('mean', V.mean(axis=0))

