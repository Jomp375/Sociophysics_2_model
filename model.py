import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.animation import FuncAnimation
# from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon
from Agent import Agent
from Door import Door

area_size_x = 50  # in meters
area_size_y = 20
danger_zone_y = area_size_y - 2
# Set the number of agents
num_blue_agents = 40
num_red_agents = 10
# Set the radius of the circle around each agent

# Constants for the forces
interaction_force = 0.10
constant_force_magnitude = 0.05
damping_coefficient = 0.5  # Damping coefficient for realistic damping force
danger_zone_force = 0.10
door_force_magnitude = 0.15
red_door_force_magnitude = 0.1

# Set the distance for the constant force towards the train door
constant_force_distance = 2.0
door_width = 2
# Set the location of the train door
door = Door(np.array([0, area_size_y]), np.array([1,0]), door_width, np.array([area_size_x * 1/2, area_size_y]), 50)
stairs_location = np.array([0, area_size_y / 2])

# Set the initial distance between agents
initial_distance_agents = 1.5  # initial minimal spread

# Set the initial velocity range of agents
initial_velocity = 0.1

# Set the time step, number of timestamps and the constant for updating positions and velocities
num_timestamps = 300
start_leaving = 100
start_entering = 150
time_step = 1


# Function to ensure agents are at least initial_distance_agents meters apart from each other
# Assuming you have 'num_agents' as the number of agents
blue_indices = [i for i in range(num_blue_agents)]
blue_agents = [Agent(i, np.array([np.random.uniform(0, area_size_x), np.random.uniform(0, area_size_y)]), (np.random.rand(2) * 2 - 1) * initial_velocity, 'Blue', (np.random.rand(1)) + 1) for i in
               blue_indices]

red_indices = [i for i in range(num_red_agents)]
red_positions = []
for i in range(num_red_agents):
    x_position = np.random.uniform(door.getposition()[0] - door_width / 2, door.getposition()[0] + door_width / 2)
    y_position = np.random.uniform(area_size_y+0.5, area_size_y + 3)
    red_positions.append([x_position, y_position])

# Create red agents with initial positions meeting the criteria
red_agents = [Agent(i, np.array(red_positions[i]), np.array([0, 0]), 'Red',3) for i in red_indices]

# Create a DataFrame to store the information
columns = ['ID', 'Time', 'X Position', 'Y Position', 'X Velocity',
           'Y Velocity', 'X Force', 'Y Force', 'Type', 'Competitiveness']
agent_data_animation = pd.DataFrame(columns=columns)
door_data_animation = pd.DataFrame(columns = ['Time','Door X Position','Door Y Position'])

# Main simulation loop
for timestamp in range(num_timestamps):
    timestamp_agent_data = pd.DataFrame(columns=columns)
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
                total_force_components += current_agent.calculate_force(agent2, door.getposition(), stairs_location)
        # Constant force towards the train door
        if current_agent.gettype() == 'Blue':
            distance_to_door = np.linalg.norm(current_agent.getposition() - door.getposition())
            if distance_to_door >= constant_force_distance:
                force_direction_to_door = (door.getposition() - current_agent.getposition()) / distance_to_door
                total_force_components += constant_force_magnitude * force_direction_to_door*current_agent.getcompetitiveness()

            # Additional force for people standing higher than y=18
            if danger_zone_y <= current_agent.getposition()[1]:
                if timestamp < start_entering or (
                        timestamp >= start_entering and (current_agent.getposition()[0] <= door.getposition()[0] - door_width / 2 or door.getposition()[0] + door_width / 2 <= current_agent.getposition()[0])):
                    total_force_components[1] -= danger_zone_force * (current_agent.getposition()[1] - danger_zone_y)

            # Force to prevent blocking the train door
            if timestamp < start_entering and door.getposition()[1]-2*door_width <= current_agent.getposition()[1]:
                if door.getposition()[0]-2*door_width/3 <= current_agent.getposition()[0] <= door.getposition()[0]:
                    door_force = door_force_magnitude * (current_agent.getposition()[0] - (
                                door.getposition()[0] - 2 * door_width / 3)) / 2
                    total_force_components[0] -= door_force

                if door.getposition()[0] < current_agent.getposition()[0] <= door.getposition()[0]+2*door_width/3:
                    door_force = door_force_magnitude * ((door.getposition()[0] + 2 * door_width / 3) - current_agent.getposition()[0]) / 2
                    total_force_components[0] += door_force

        elif current_agent.gettype() == 'Red':
            if timestamp > start_leaving:
                # Force for going towards the stairs
                distance_to_stairs = np.linalg.norm(current_agent.getposition() - stairs_location)
                force_direction_to_stairs = (stairs_location - current_agent.getposition()) / distance_to_stairs
                total_force_components += 5/2*constant_force_magnitude * force_direction_to_stairs
            else:
                distance_to_door = np.linalg.norm(current_agent.getposition() - door.getposition())
                # Force to go stand in front of the train door
                if distance_to_door > 0.8:
                    force_direction_to_door = (door.getposition() - current_agent.getposition()) / distance_to_door
                    total_force_components += 3*constant_force_magnitude * force_direction_to_door
            # Force to go through the train door
            if door.getposition()[1]-2*door_width <= current_agent.getposition()[1]:
                if current_agent.getposition()[0] <= door.getposition()[0]:
                    door_force = 2*red_door_force_magnitude * (-current_agent.getposition()[0] + door.getposition()[0])
                    total_force_components[0] += door_force

                if door.getposition()[0] < current_agent.getposition()[0]:
                    door_force = 2*red_door_force_magnitude * (door.getposition()[0] - current_agent.getposition()[0])
                    total_force_components[0] += door_force

        # Calculate the net force magnitude
        net_force_magnitude = np.linalg.norm(total_force_components)

        # resistance force to prevent large velocities
        resistance_force = -damping_coefficient * current_agent.getvelocity()
        agent_speed = np.linalg.norm(current_agent.getvelocity())
        if agent_speed > current_agent.getmaxvelocity():
            resistance_force *= agent_speed / current_agent.getmaxvelocity()
        total_force_components += resistance_force

        # Introduce opposing force when net force magnitude is less than 0.5
        if (net_force_magnitude < 0.09) & (current_agent.getvelocity().all() < 0.4):
            # Calculate opposing force as the opposite of the net force
            opposing_force = -total_force_components - (current_agent.getvelocity() / time_step)

            # Add the opposing force to total force components
            total_force_components += opposing_force

        # Reset forces and set velocities for agents above area_size after time 150
        if timestamp >= start_entering and current_agent.getposition()[1] > area_size_y and current_agent.gettype() == 'Blue':
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
            'Type': current_agent.gettype(),
            'Competitiveness': current_agent.getcompetitiveness(),
        }, index=blue_indices)

        timestamp_agent_data = pd.concat([timestamp_agent_specific_data, timestamp_agent_data], ignore_index=True)

    timestamp_Door_data = pd.DataFrame( {'Time': timestamp + 1,
                                        'Door X Position': door.getposition()[0],
                                        'Door Y Position': door.getposition()[1]}, index=np.array([1]))
    door.update_velocity(time_step)
    door.update_position(time_step)
    door_data_animation = pd.concat([door_data_animation,timestamp_Door_data],ignore_index=True)
    agent_data_animation = pd.concat([agent_data_animation, timestamp_agent_data], ignore_index=True)


def update(frame):
    plt.clf()  # Clear the previous plot

    # Plot agents
    agent_data_frame = agent_data_animation[agent_data_animation['Time'] == frame]
    door_data_frame = door_data_animation[door_data_animation['Time'] == frame]
    # Define a colormap based on competitiveness
    cmap = cm.get_cmap('viridis')  # You can change the colormap here
    competitiveness_values = agent_data_frame['Competitiveness']
    norm = plt.Normalize(competitiveness_values.min(), competitiveness_values.max())

    # Plot agents with colors based on competitiveness
    plt.scatter(
        agent_data_frame['X Position'],
        agent_data_frame['Y Position'],
        label='Agent Positions',
        marker='o',
        c=competitiveness_values,
        cmap=cmap,
        norm=norm,
        s=area_size_y
    )
    plt.colorbar(label='Competitiveness')  # Add a color bar

    # Add a marker as a train door
    plt.scatter(door_data_frame['Door X Position'], door_data_frame['Door Y Position'], marker='o', color='orange', s=200, label='Train Door')

    # Add a marker as the stairs
    plt.scatter(stairs_location[0], stairs_location[1], marker='s', color='black', s=200, label='Stairs')

    # Draw the train door box
    # door_vertices = np.array(
    #     [(door_data_frame['Door X Position'] - door_width / 2, door_data_frame['Door Y Position']),
    #     (door_data_frame['Door X Position'] + door_width / 2, door_data_frame['Door Y Position']),
    #     (door_data_frame['Door X Position'] + door_width / 2, door_data_frame['Door Y Position'] - 1),
    #     (door_data_frame['Door X Position'] - door_width / 2, door_data_frame['Door Y Position'] - 1)])
    # door_box = Polygon(door_vertices, edgecolor='blue', facecolor='none')
    # plt.gca().add_patch(door_box)

    # Set fixed axes limits
    plt.xlim(0, area_size_x)
    plt.ylim(0, area_size_y + 1)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Time = {frame}')
    plt.grid(True)


# Set up the figure and axis
fig, ax = plt.subplots()

# Get unique time values from the DataFrame
unique_times = agent_data_animation['Time'].unique()

# Create the animation
animation = FuncAnimation(fig, update, frames=unique_times, interval=50, repeat=True)
# To save the animation using Pillow as a gif
# writer = PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
# animation.save('boarding_animation.gif', writer=writer)
# Display the animation
plt.show()
