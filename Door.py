import numpy as np



class Door:
    def __init__(self, time_X, position_at_time_X, final_position, final_time, door_width):
        self.final_position = np.array(final_position)
        self.final_time = final_time
        self.door_width = door_width

        # Calculate initial velocity to ensure the door is standing still at the final time and position
        self.initial_velocity = np.divide((final_position - position_at_time_X), (final_time - time_X))

        # Calculate initial position
        self.initial_position = position_at_time_X - time_X * self.initial_velocity

        self.position = np.array(self.initial_position)
        self.velocity = np.array(self.initial_velocity)

        # Calculate acceleration
        self.acceleration = 2 * np.divide(
            (self.final_position - self.initial_position - self.initial_velocity * self.final_time),
            (self.final_time ** 2))
        print(self.acceleration)

    def update_position(self, timestep):
        self.position = self.position.astype(np.float64)
        if np.linalg.norm(self.position - self.final_position) < 0.02:
            self.position = self.final_position
        else:
             self.position += timestep * self.velocity

    def update_velocity(self, timestep):
        self.velocity = self.velocity.astype(np.float64)
        if np.linalg.norm(self.position - self.final_position) < 0.02:
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
