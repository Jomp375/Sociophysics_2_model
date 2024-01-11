import numpy as np


class Door:
    def __init__(self, initial_position, initial_velocity, door_width, final_position, final_time):
        self.initial_position = np.array(initial_position)
        self.position = np.array(initial_position)
        self.initial_velocity = np.array(initial_velocity)
        self.velocity = self.initial_velocity
        self.door_width = door_width
        self.final_position = np.array(final_position)
        self.final_time = final_time
        self.acceleration = 2 * np.divide(
            (self.final_position - self.initial_position - self.initial_velocity * self.final_time),
            (self.final_time ** 2))

    def update_position(self, timestep):
        self.position = self.position.astype(np.float64)
        if np.linalg.norm(self.position - self.final_position) < 0.3:
            self.position = self.final_position
        else:
            self.position += timestep * self.velocity

    def update_velocity(self, timestep):
        self.velocity = self.velocity.astype(np.float64)
        if np.linalg.norm(self.position - self.final_position) < 0.3:
            self.velocity = np.array([0, 0])
        else:
            self.velocity += timestep * self.acceleration

    def getposition(self):
        return self.position

    def getvelocity(self):
        return self.velocity

    def setposition(self, position):
        self.position = position
        pass

    def setvelocity(self, velocity):
        self.velocity = velocity
        pass
