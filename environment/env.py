from environment.carla.client import make_carla_client
from environment.carla.settings import CarlaSettings
from environment.carla.sensor import Camera
from environment.carla.carla_server_pb2 import Control
from environment.imitation.imitation_learning import ImitationLearning
import random
import time
import os
from PIL import Image
import numpy as np


class action_space(object):
    def __init__(self, dim, high, low, seed):
        self.shape = (dim,)
        self.high = np.array(high)
        self.low = np.array(low)
        self.seed = seed
        assert(dim == len(high) == len(low))
        np.random.seed(self.seed)

    def sample(self):
        return np.random.uniform(self.low + (0, 0.8), self.high)

class observation_space(object):
    def __init__(self, dim, high=None, low=None, seed=None):
        self.shape = (dim,)
        self.high = high
        self.low = low
        self.seed = seed


class Env(object):
    def __init__(self, MONITOR_DIR, SEED, FPS, sess, action_lambda=0.5):
        self.MONITOR_DIR = MONITOR_DIR
        self.client = None
        self.Image_agent = ImitationLearning(sess)
        self.action_space = action_space(2, (1.0, 1.0), (-1.0, -1.0), SEED)
        self.observation_space = observation_space(512 + 3 + 2)
        self.render_ = False
        self.action_lambda = action_lambda
        self.image_dir_ = None
        self.image_i_ = 0
        self.FPS = FPS
        self.reward_time = 0
        self.prev_measurements = None
        self.prev_action = {'steer': 0.0, 'acc': 0.0, 'brake': 0.0}

    def connected(self, client):
        self.render_ = False
        self.client = client
        self.reward_time = 0

    def step(self, action):
        steer = action['steer']
        acc = action['acc']
        brake = action['brake']
        control = Control()
        control.steer = steer * self.action_lambda + (1 - self.action_lambda) * self.prev_action['steer']
        control.throttle = acc * self.action_lambda + (1 - self.action_lambda) * self.prev_action['acc']
        control.brake = brake * self.action_lambda + (1 - self.action_lambda) * self.prev_action['brake']
        control.hand_brake = 0
        control.reverse = 0
        if self.prev_measurements is not None and self.prev_measurements.player_measurements.forward_speed >= 8:
            control.throttle = 0.5 if control.throttle > 0.5 else control.throttle
        self.client.send_control(control)
        measurements, sensor_data = self.client.read_data()
        if self.render_:
            im = sensor_data['CameraRGB'].data[115:510, :]
            im = Image.fromarray(im)
            if not os.path.isdir(os.path.join(self.MONITOR_DIR, self.image_dir_)):
                os.makedirs(os.path.join(self.MONITOR_DIR, self.image_dir_))
            im.save(os.path.join(self.MONITOR_DIR, self.image_dir_, str(self.image_i_) + '.jpg'))
            self.image_i_ += 1

        feature_vector = self.Image_agent.compute_feature(sensor_data)
        speed = measurements.player_measurements.forward_speed
        speed = speed / 10.0
        offroad = measurements.player_measurements.intersection_offroad
        other_lane = measurements.player_measurements.intersection_otherlane

        reward, done = self.reward(measurements, self.prev_measurements, action, self.prev_action)

        self.prev_action['steer'] = control.steer
        self.prev_action['acc'] = control.throttle
        self.prev_action['brake'] = control.brake

        self.prev_measurements = measurements
        info = 0

        return np.concatenate((feature_vector, (control.steer, control.throttle - control.brake, speed, offroad, other_lane))), reward, done, info

    def reset(self):
        print('start to reset env')
        self.image_i_ = 0
        self.image_dir_ = None
        self.render_ = False
        self.reward_time = 0
        self.prev_measurements = None
        self.prev_action = {'steer': 0.0, 'acc': 0.0, 'brake': 0.0}
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
        observation = None

        scene = self.client.load_settings(settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = random.randint(0, max(0, number_of_player_starts - 1))
        self.client.start_episode(player_start)
        for i in range(20):
            action = {'steer': 0.0, 'acc': 0.0, 'brake': 0.0}
            observation, _, _, _ = self.step(action)
            time.sleep(0.1)
        self.reward_time = 0
        print('reset finished')
        return observation

    def render(self, image_dir):
        self.render_ = True
        self.image_dir_ = image_dir

    def reward(self, measurements, prev_measurements, action, prev_action):
        """
        :param measurements:
        :param prev_measurements: due to the bug in the carla platform, we need to use this ugly
                                  parameter to alarm collision in low speed.
        :param steer:
        :return: reward, done
        """
        done = False
        reward = 0.0

        """collision"""
        if measurements.player_measurements.collision_other != 0 or \
                measurements.player_measurements.collision_pedestrians != 0 or \
                measurements.player_measurements.collision_vehicles != 0:
            reward = reward - 30
            done = True

        """road"""
        reward = reward - 10 * (measurements.player_measurements.intersection_offroad +
                               measurements.player_measurements.intersection_otherlane)

        #if reward < -12:
        #   done = True

        actual_steer = action['steer'] * self.action_lambda + (1 - self.action_lambda) * prev_action['steer']

        if measurements.player_measurements.forward_speed <= 6:
            reward = reward + measurements.player_measurements.forward_speed**2 / 6.0
        elif measurements.player_measurements.forward_speed <= 8:
            reward = reward + 3 * (8 - measurements.player_measurements.forward_speed)
        else:
            reward = reward - 2 * (measurements.player_measurements.forward_speed - 8) ** 2

        reward = reward - 4 * np.abs(actual_steer) * np.abs(actual_steer) * \
                 measurements.player_measurements.forward_speed

        if prev_measurements is None:
            return reward, done
        x, y = measurements.player_measurements.transform.location.x, \
               measurements.player_measurements.transform.location.y
        prev_x, prev_y = prev_measurements.player_measurements.transform.location.x, \
                         prev_measurements.player_measurements.transform.location.y
        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)

        if distance < 1.0 / self.FPS * 1 or measurements.player_measurements.forward_speed < 0:
            reward = reward - 2
            if measurements.player_measurements.forward_speed < 0:
                self.reward_time += 5
            self.reward_time += 1
        else:
            self.reward_time = 0

        if self.reward_time >= 20:
            done = True
            self.reward_time = 0

        return reward, done

