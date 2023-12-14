import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


class Agent:
    def __init__(self, id, initial_position, initial_velocity, type, ):
        self.id = id
        self.position = initial_position
        self.velocity = initial_velocity
        self.type = type
        self.competitiveness = None  # Example of additional variable

    def calculate_force(self, other_agent, door_location, stairs_location):
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
        if self.type == 'Blue' and distance_to_door_agent_i > distance_to_door_agent_j and distance < circle_radius:
            force_magnitude = interaction_force * np.square((1 - distance / circle_radius))
        elif (self.type == 'Red' and distance_to_stairs_agent_i > distance_to_stairs_agent_j and
              distance < red_circle_radius):
            force_magnitude = interaction_force * np.square((1 - distance / red_circle_radius))
            # Calculate force direction (away from the agent providing the force)
        force_direction = -distance_vector / distance

        # Accumulate force components
        total_force = force_magnitude * force_direction
        return total_force  # Convert if necessary
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


# Set the size of the area
area_size = 20  # in meters

dangerzoney = area_size - 2
# Set the number of agents
num_blue_agents = 20
num_red_agents = 10
circle_radius = 2.5
red_circle_radius = 1.5
# Set the radius of the circle around each agent


interaction_force = 0.12
constant_force_magnitude = 0.07
damping_coefficient = 0.4  # Damping coefficient for realistic damping force
dangerzone_force = 0.10
# Set the location of the train door
door_location = np.array([area_size / 2, area_size])
stairs_location = np.array([0,area_size / 2])
door_width = 2
# Set the distance for the constant force towards the train door
constant_force_distance = 2.0

# Set the initial distance between agents
initial_distance_agents = 1.5  # initial minimal spread

# Set the initial velocity range of agents
initial_velocity = 0.1

# Set the time step, number of timestamps and the constant for updating positions and velocities
num_timestamps = 300
start_leaving = 100
start_entering = 150
time_step = 1

max_velocity = 1

# Additional force for people standing between y=18 and y=20

# Force to prevent blocking the train door
door_force_magnitude = 0.15
red_door_force_magnitude = 0.1
# Function to ensure agents are at least initial_distance_agents meters apart from each other
# Assuming you have 'num_agents' as the number of agents
blue_indices = [i for i in range(num_blue_agents)]
blue_agents = [Agent(i, np.random.rand(2) * area_size, (np.random.rand(2) * 2 - 1) * initial_velocity, 'Blue') for i in
               blue_indices]

red_indices = [i for i in range(num_red_agents)]
red_positions = []
for i in range(num_red_agents):
    x_position = np.random.uniform(door_location[0] - door_width/2, door_location[0] + door_width/2)
    y_position = np.random.uniform(area_size+0.5, area_size +3)
    red_positions.append([x_position, y_position])

# Create red agents with initial positions meeting the criteria
red_agents = [Agent(i, np.array(red_positions[i]), np.array([0, 0]), 'Red') for i in red_indices]

# Create a DataFrame to store the information
columns = ['ID', 'Time', 'X Position', 'Y Position', 'X Velocity', 'Y Velocity', 'X Force', 'Y Force', 'Type']
agent_data_animatie = pd.DataFrame(columns=columns)

# Main simulation loop
for timestamp in range(num_timestamps):
    timestamp_agent_data = pd.DataFrame(columns=['ID', 'Time', 'X Position', 'Y Position',
                                                 'X Velocity', 'Y Velocity', 'X Force',
                                                 'Y Force', 'Type'])
    if timestamp < start_entering:
        constant_force_distance = 2.4  # Use the original value before time 150
    else:
        constant_force_distance = 0  # Set to 0 after time 150
    for i in range(num_blue_agents + num_red_agents):
        if i < num_blue_agents:
            current_agent = blue_agents[i]
        else:
            current_agent = red_agents[i - num_blue_agents]

        total_force_components = np.zeros(2)  # Reset total force components

        for j in range(num_blue_agents + num_red_agents):
            if i != j:
                if j < num_blue_agents:
                    agent2 = blue_agents[j]
                else:
                    agent2 = red_agents[j-num_blue_agents]
                # apply forces with each other
                total_force_components += current_agent.calculate_force(agent2, door_location, stairs_location)
        # Constant force towards the train door
        if current_agent.gettype() == 'Blue':
            distance_to_door = np.linalg.norm(current_agent.getposition() - door_location)
            if distance_to_door >= constant_force_distance:
                force_direction_to_door = (door_location - current_agent.getposition()) / distance_to_door
                total_force_components += constant_force_magnitude * force_direction_to_door

            # Additional force for people standing higher than y=18
            if dangerzoney <= current_agent.getposition()[1]:
                if timestamp < start_entering or (
                        timestamp >= start_entering and (current_agent.getposition()[0] <= door_location[0] - door_width / 2 or door_location[0] + door_width / 2 <= current_agent.getposition()[0])):
                    total_force_components[1] -= dangerzone_force * (current_agent.getposition()[1] - dangerzoney)

            # Force to prevent blocking the train door
            if timestamp < start_entering and door_location[1]-3*door_width <= current_agent.getposition()[1]:
                if door_location[0]-2*door_width/3 <= current_agent.getposition()[0] <= door_location[0]:
                    door_force = door_force_magnitude * (current_agent.getposition()[0] - (door_location[0] - 2 * door_width / 3)) / 2
                    total_force_components[0] -= door_force

                if door_location[0] < current_agent.getposition()[0] <= door_location[0]+2*door_width/3:
                    door_force = door_force_magnitude * ((door_location[0]+2*door_width/3) - current_agent.getposition()[0]) / 2
                    total_force_components[0] += door_force

        elif current_agent.gettype() == 'Red':
            if timestamp > start_leaving:
                # Force for going towards the stairs
                distance_to_stairs = np.linalg.norm(current_agent.getposition() - stairs_location)
                force_direction_to_stairs = (stairs_location - current_agent.getposition()) / distance_to_stairs
                total_force_components += constant_force_magnitude * force_direction_to_stairs
            else:
                distance_to_door = np.linalg.norm(current_agent.getposition() - door_location)
                # Force to go stand in front of the train door
                if distance_to_door > 0.8:
                    force_direction_to_door = (door_location - current_agent.getposition()) / distance_to_door
                    total_force_components += 2*constant_force_magnitude * force_direction_to_door
            # Force to go through the train door
            if door_location[1]-door_width <= current_agent.getposition()[1]:
                if current_agent.getposition()[0] <= door_location[0]:
                    door_force = 2*red_door_force_magnitude * (-current_agent.getposition()[0] + door_location[0])
                    total_force_components[0] += door_force

                if door_location[0] < current_agent.getposition()[0]:
                    door_force = 2*red_door_force_magnitude * (door_location[0] - current_agent.getposition()[0])
                    total_force_components[0] += door_force

        # Calculate the net force magnitude
        net_force_magnitude = np.linalg.norm(total_force_components)

        # resistance force to prevent large velocities
        resistance_force = -damping_coefficient * current_agent.getvelocity()
        agent_speed = np.linalg.norm(current_agent.getvelocity())
        if agent_speed > max_velocity:
            resistance_force *= agent_speed / max_velocity
        total_force_components += resistance_force

        # Introduce opposing force when net force magnitude is less than 0.5
        if (net_force_magnitude < 0.09) & (current_agent.getvelocity().all() < 0.4):
            # Calculate opposing force as the opposite of the net force
            opposing_force = -total_force_components - (current_agent.getvelocity() / time_step)

            # Add the opposing force to total force components
            total_force_components += opposing_force

        # Reset forces and set velocities for agents above area_size after time 150
        if timestamp >= start_entering and current_agent.getposition()[1] > area_size and current_agent.gettype()== 'Blue':
            total_force_components = np.zeros(2)  # Reset forces
            current_agent.setvelocity(np.array([0.0, 0.2]))  # Set the desired velocity
        # Update positions, velocities, and forces for each agent
        current_agent.update_position(time_step)
        current_agent.update_velocity(time_step, total_force_components)
        # Store the information for each agent at the current timestamp
        timestamp_agent_specific_data = pd.DataFrame({
            'ID': current_agent.getid(),
            'Time': timestamp + 1,
            'X Position': current_agent.getposition()[0],
            'Y Position': current_agent.getposition()[1],
            'X Velocity': current_agent.getvelocity()[0],
            'Y Velocity': current_agent.getvelocity()[1],
            'X Force': total_force_components[0],
            'Y Force': total_force_components[1],
            'Type': current_agent.gettype()
        }, index=blue_indices)

        timestamp_agent_data = pd.concat([timestamp_agent_specific_data, timestamp_agent_data], ignore_index=True)

    agent_data_animatie = pd.concat([agent_data_animatie, timestamp_agent_data], ignore_index=True)


def update(frame):
    plt.clf()  # Clear the previous plot

    # Plot agents
    agent_data_frame = agent_data_animatie[agent_data_animatie['Time'] == frame]
    plt.scatter(agent_data_frame['X Position'], agent_data_frame['Y Position'], label='Agent Positions',marker='o',color=agent_data_frame['Type'], s=area_size)

    # Add a marker as a train door
    plt.scatter(door_location[0],door_location[1], marker='o', color='orange', s=200, label='Train Door')

    # Add a marker as the stairs
    plt.scatter(stairs_location[0],stairs_location[1], marker='s', color='Black', s=200, label='Stairs')

    # Draw the train door box
    door_vertices = np.array(
        [(area_size / 2 - 1, area_size - 1), (area_size / 2 + 1, area_size - 1), (area_size / 2 + 1, area_size),
         (area_size / 2 - 1, area_size)])
    door_box = Polygon(door_vertices, edgecolor='blue', facecolor='none')
    plt.gca().add_patch(door_box)

    # Set labels and show the plot
    plt.xlim(0, area_size)
    plt.ylim(0, area_size + 1)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Time = {frame}')
    plt.grid(True)

# Set up the figure and axis
fig, ax = plt.subplots()

# Get unique time values from the DataFrame
unique_times = agent_data_animatie['Time'].unique()

# Create the animation
animation = FuncAnimation(fig, update, frames=unique_times, interval=50, repeat=True)
#animation.convert_path: 'C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\convert.exe'
#animation.save('animation.gif', writer='imagemagick', fps=30)
# Display the animation
plt.show()