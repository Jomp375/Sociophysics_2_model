import numpy as np
import random

class Agent:
    def __init__(self, id: int, initial_position, initial_velocity, type: str, competitiveness: float, frustration: float):
        self.id = id
        self.position = initial_position
        self.velocity = initial_velocity
        self.type = type
        self.competitiveness = competitiveness  # Example of additional variable
        self.max_velocity = np.random.rand(1) + self.competitiveness
        self.frustration = frustration

    def calculate_force(self, other_agent, door_location, stairs_location):
        interaction_force = 0.10
        circle_radius = 2.5
        red_circle_radius = 1.5
        total_force = np.zeros(2)
        force_magnitude = 0
        # Implement the force calculation method for the agent
        # Calculate the distance between agents
        distance_vector = other_agent.getposition() - self.position
        distance = abs(np.linalg.norm(distance_vector))
        # Check if the agent is farther from the door than a threshold before applying interaction force
        distance_to_door_agent_i = np.linalg.norm(self.position - door_location)
        distance_to_door_agent_j = np.linalg.norm(other_agent.getposition() - door_location)
        distance_to_stairs_agent_i = np.linalg.norm(self.position - stairs_location)
        distance_to_stairs_agent_j = np.linalg.norm(other_agent.getposition() - stairs_location)
        if (self.type == 'Blue' and distance_to_door_agent_i > distance_to_door_agent_j and
                distance < (circle_radius - self.frustration)):
            force_magnitude = interaction_force * np.square((1 - distance / circle_radius))
        elif (self.type == 'Red' and distance_to_stairs_agent_i > distance_to_stairs_agent_j and
              distance < (red_circle_radius - self.frustration)):
            force_magnitude = interaction_force * np.square((1 - distance / red_circle_radius))
            # Calculate force direction (away from the agent providing the force)
        force_direction = -distance_vector / distance

        # Accumulate force components
        total_force += force_magnitude * force_direction
        return total_force
        pass

    def update_position(self, timestep):
        # Implement method to update agent's position
        self.position += timestep * self.velocity
        pass

    def update_velocity(self, timestep, total_force):
        # Convert self.velocity to float64 if it's of integer type
        self.velocity = self.velocity.astype(np.float64)
        # Implement method to update agent's velocity based on total force
        self.velocity += timestep * total_force
        pass

    def update_frustration(self, timestep):
        if self.frustration < 2 and random.random() < 0.02:
            self.frustration += 0.03 * timestep
        pass

    def getid(self):
        return self.id

    def gettype(self):
        return self.type

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

    def getmaxvelocity(self):
        return self.max_velocity
        pass

    def getcompetitiveness(self):
        return self.competitiveness
        pass

    def getfrustration(self):
        return self.frustration
        pass
