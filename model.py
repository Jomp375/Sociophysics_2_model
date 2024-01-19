import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.animation import FuncAnimation
# from matplotlib.animation import PillowWriter
from Agent import Agent
from Door import Door

area_size_x = 50  # in meters
area_size_y = 20
danger_zone_y = area_size_y - 2
# Set the number of agents
num_blue_agents = 40
num_red_agents = 10
# Set the radius of the circle around each agent
pole_radius = 1


def simulation(Blue_agents, Red_agents, train_door: Door, timestep: int, delay:float):
    num_timestamps = round((750 + delay) / timestep)
    start_leaving = 135 + delay
    start_entering = 250 + delay
    stopping_time = 120 + delay
    arrival_time = 100 + delay
    stairs_location = np.array([0, area_size_y / 2])
    constant_force_door_magnitude = 0.05
    constant_force_up_magnitude = 0.03
    damping_coefficient = 0.5  # Damping coefficient for realistic damping force
    danger_zone_force = 0.10
    door_force_magnitude = 0.15
    red_door_force_magnitude = 0.1
    # Create a DataFrame to store the information
    columns = ['ID', 'Time', 'X Position', 'Y Position', 'X Velocity',
               'Y Velocity', 'X Force', 'Y Force', 'Type', 'Competitiveness', 'Frustration']
    agent_data_animation = pd.DataFrame(columns=columns)
    door_data_animation = pd.DataFrame(columns=['Time', 'Door X Position', 'Door Y Position'])
    nr_blue_agents = len(Blue_agents)
    nr_red_agents = len(Red_agents)
    # Main simulation loop
    for timestamp in range(num_timestamps):
        pole_location = np.array(train_door.getposition()[0], train_door.getposition()[1] - 0.8)
        timestamp_agent_data = pd.DataFrame(columns=columns)
        if timestamp*timestep < start_entering:
            distance_for_force = 2.4  # Use the original value before time 150
        else:
            distance_for_force = 0  # Set to 0 after time 150
        for i in range(nr_blue_agents + nr_red_agents):
            if i < nr_blue_agents:
                current_agent = Blue_agents[i]
            else:
                current_agent = Red_agents[i - nr_blue_agents]

            total_force_components = np.zeros(2)  # Reset total force components

            for j in range(nr_blue_agents + num_red_agents):
                if i != j:
                    if j < nr_blue_agents:
                        agent2 = Blue_agents[j]
                    else:
                        agent2 = Red_agents[j - nr_blue_agents]
                    # apply forces with each other
                    total_force_components += current_agent.calculate_force(agent2, train_door.getposition(), stairs_location)
            # Constant force towards the train door
            if current_agent.gettype() == 'Blue':
                distance_to_door = np.linalg.norm(current_agent.getposition() - train_door.getposition())
                if timestamp*timestep < arrival_time:
                    total_force_components += constant_force_up_magnitude * np.array([0, 1])
                elif distance_to_door >= distance_for_force:
                    force_direction_to_door = (train_door.getposition() - current_agent.getposition()) / distance_to_door
                    total_force_components += constant_force_door_magnitude * force_direction_to_door * current_agent.getcompetitiveness()

                # Additional force for people standing higher than y=18
                if danger_zone_y <= current_agent.getposition()[1]:
                    if timestamp*timestep < start_entering or (
                            timestamp * timestep >= start_entering and (
                            current_agent.getposition()[0] <= train_door.getposition()[0] - train_door.getwidth() / 2 or train_door.getposition()[
                        0] + train_door.getwidth() / 2 <= current_agent.getposition()[0])):
                        total_force_components[1] -= danger_zone_force * (current_agent.getposition()[1] - danger_zone_y)

                # Force to prevent blocking the train door
                if (stopping_time < timestamp*timestep < start_entering
                        and train_door.getposition()[1] - 2 * train_door.getwidth() <= current_agent.getposition()[1]):
                    if train_door.getposition()[0] - 2 * train_door.getwidth() / 3 <= current_agent.getposition()[0] <= train_door.getposition()[
                        0]:
                        door_force = door_force_magnitude * (current_agent.getposition()[0] - (
                                train_door.getposition()[0] - 2 * train_door.getwidth() / 3)) / 2
                        total_force_components[0] -= door_force

                    if train_door.getposition()[0] < current_agent.getposition()[0] <= train_door.getposition()[0] + 2 * train_door.getwidth() / 3:
                        door_force = door_force_magnitude * (
                                (train_door.getposition()[0] + 2 * train_door.getwidth() / 3) - current_agent.getposition()[0]) / 2
                        total_force_components[0] += door_force

            elif current_agent.gettype() == 'Red':
                if timestamp*timestep > start_leaving:
                    if current_agent.getposition()[1] < train_door.getposition()[1] - 2 * train_door.getwidth():
                        # Force for going towards the stairs
                        distance_to_stairs = np.linalg.norm(current_agent.getposition() - stairs_location)
                        force_direction_to_stairs = (stairs_location - current_agent.getposition()) / distance_to_stairs
                        total_force_components += 5 / 2 * constant_force_door_magnitude * force_direction_to_stairs
                    else:
                        total_force_components += np.array([0,-constant_force_up_magnitude])
                else:
                    distance_to_door = np.linalg.norm(current_agent.getposition() - train_door.getposition())
                    # Force to go stand in front of the train door
                    if distance_to_door > 0.8:
                        force_direction_to_door = (train_door.getposition() - current_agent.getposition()) / distance_to_door
                        total_force_components += 2 * constant_force_door_magnitude * force_direction_to_door

                # Force to go through the train door
                if train_door.getposition()[1] - 2 * train_door.getwidth() <= current_agent.getposition()[1]:
                    if current_agent.getposition()[0] <= train_door.getposition()[0]:
                        door_force = 2 * red_door_force_magnitude * (
                                -current_agent.getposition()[0] + train_door.getposition()[0])
                        total_force_components[0] += door_force

                    if train_door.getposition()[0] < current_agent.getposition()[0]:
                        door_force = 2 * red_door_force_magnitude * (train_door.getposition()[0] - current_agent.getposition()[0])
                        total_force_components[0] += door_force

            # Force that prevents people from bumping into the pole

            # vector_to_pole = Pole_location - current_agent.getposition()
            # distance = abs(np.linalg.norm(vector_to_pole))
            # if distance < pole_radius:
            #     force_magnitude = interaction_force * np.square((1 - distance / pole_radius))
            # else:
            #     force_magnitude = 0
            # force_direction = vector_to_pole / distance
            # total_force_components += force_magnitude * force_direction
            # Calculate the net force magnitude

            net_force_magnitude = np.linalg.norm(total_force_components)

            # resistance force to prevent large velocities
            resistance_force = -damping_coefficient * current_agent.getvelocity()
            agent_speed = np.linalg.norm(current_agent.getvelocity())
            if agent_speed > current_agent.getmaxvelocity()*timestep:
                resistance_force *= agent_speed / current_agent.getmaxvelocity()
            total_force_components += resistance_force

            # Introduce opposing force when net force magnitude is less than 0.5
            if (net_force_magnitude < 0.03) & (current_agent.getvelocity().all() * timestep < 0.3):
                # Calculate opposing force as the opposite of the net force
                opposing_force = -total_force_components - (current_agent.getvelocity() / timestep)

                # Add the opposing force to total force components
                total_force_components += opposing_force

            # Reset forces and set velocities for agents above area_size after time 150
            if timestamp*timestep >= start_entering and current_agent.getposition()[
                1] > area_size_y and current_agent.gettype() == 'Blue':
                total_force_components = np.zeros(2)  # Reset forces
                current_agent.setvelocity(np.array([0.0, 0.2]))  # Set the desired velocity
            # Update positions, velocities, and forces for each agent
            current_agent.update_position(timestep)
            current_agent.update_velocity(timestep, total_force_components)
            current_agent.update_frustration(timestep)
            # Store the information for each agent at the current timestamp
            timestamp_agent_specific_data = pd.DataFrame({
                'ID': current_agent.getid(),
                'Time': timestamp * timestep + 1,
                'X Position': current_agent.getposition()[0],
                'Y Position': current_agent.getposition()[1],
                'X Velocity': current_agent.getvelocity()[0],
                'Y Velocity': current_agent.getvelocity()[1],
                'X Force': total_force_components[0],
                'Y Force': total_force_components[1],
                'Type': current_agent.gettype(),
                'Competitiveness': current_agent.getcompetitiveness(),
                'Frustration': current_agent.getfrustration()
            }, index=[current_agent.getid])

            timestamp_agent_data = pd.concat([timestamp_agent_specific_data, timestamp_agent_data], ignore_index=True)

        timestamp_Door_data = pd.DataFrame({'Time': timestamp * timestep + 1,
                                            'Door X Position': train_door.getposition()[0],
                                            'Door Y Position': train_door.getposition()[1]}, index=np.array([1]))
        train_door.update_velocity(timestep)
        train_door.update_position(timestep)
        door_data_animation = pd.concat([door_data_animation, timestamp_Door_data], ignore_index=True)
        agent_data_animation = pd.concat([agent_data_animation, timestamp_agent_data], ignore_index=True)
    return agent_data_animation, door_data_animation


def testing(nr_of_runs:int, delay_increase: float):
    result_df = pd.DataFrame(columns=['Delay', 'Boarding Time'])
    for i in range(1,nr_of_runs):
        time_step = 1
        delay = i*delay_increase
        stopping_time = 120 + delay
        arrival_time = 100 + delay
        door_width = 1.8
        # Set the location of the train door
        door = Door(arrival_time, np.array([area_size_x * 1 / 4, area_size_y]),
                    np.array([area_size_x * 1 / 2, area_size_y]),
                    stopping_time, door_width)
        # Set the initial velocity range of agents
        initial_velocity = 0.1

        # Function to ensure agents are at least initial_distance_agents meters apart from each other
        # Assuming you have 'num_agents' as the number of agents
        blue_indices = [i for i in range(num_blue_agents)]
        blue_agents = [Agent(i, np.array([np.random.uniform(0, area_size_x),
                                          np.random.uniform(area_size_y / 4, danger_zone_y)]),
                             (np.random.rand(2) * 2 - 1) * initial_velocity,
                             'Blue', (np.random.rand(1)) + 1, np.random.uniform(0, 0.5)) for i in blue_indices]
        red_indices = [i for i in range(num_red_agents)]
        red_positions = []
        for i in range(num_red_agents):
            x_position = np.random.uniform(door.getposition()[0] - door_width / 2,
                                           door.getposition()[0] + door_width / 2)
            y_position = np.random.uniform(area_size_y + 0.5, area_size_y + 3)
            red_positions.append([x_position, y_position])
        # Create red agents with initial positions meeting the criteria
        red_agents = [Agent(i, np.array(red_positions[i]), np.array([0, 0]), 'Red', 3, 0) for i in red_indices]
        sim_result = simulation(blue_agents, red_agents,door, time_step, delay)[0]
        eadz = sim_result[sim_result['Y position'] > danger_zone_y & sim_result['Type'] == 'Blue']
        id_counts = eadz.groupby('Time')['ID'].nunique()
        earliest_time = id_counts[id_counts == num_blue_agents].index.min()
        run_data = pd.DataFrame({'Delay': delay,
            'Boarding Time': earliest_time}, index=[i])

        result_df = pd.concat([run_data, result_df], ignore_index=True)
    return result_df


delay = 0
time_step = 1
num_timestamps = round((650 + delay) / time_step)
stopping_time = 120 + delay
arrival_time = 100 + delay
door_width = 1.8
# Set the location of the train door
door = Door(arrival_time, np.array([area_size_x * 1 / 4, area_size_y]),
                    np.array([area_size_x * 1 / 2, area_size_y]),
                    stopping_time, door_width)
stairs_location = np.array([0, area_size_y / 2])
# Set the initial velocity range of agents
initial_velocity = 0.1
# Assuming you have 'num_agents' as the number of agents
blue_indices = [i for i in range(num_blue_agents)]
blue_agents = [Agent(i, np.array([np.random.uniform(0, area_size_x),
                                          np.random.uniform(area_size_y / 4, danger_zone_y)]),
                             (np.random.rand(2) * 2 - 1) * initial_velocity,
                             'Blue', (np.random.rand(1)) + 1, np.random.uniform(0, 0.5)) for i in blue_indices]
red_indices = [i for i in range(num_red_agents)]
red_positions = []
for i in range(num_red_agents):
    x_position = np.random.uniform(door.getposition()[0] - door_width / 2,
                                           door.getposition()[0] + door_width / 2)
    y_position = np.random.uniform(area_size_y + 0.5, area_size_y + 3)
    red_positions.append([x_position, y_position])
# Create red agents with initial positions meeting the criteria
red_agents = [Agent(i, np.array(red_positions[i]), np.array([0, 0]), 'Red', 3, 0) for i in red_indices]


ran_simulation = simulation(blue_agents, red_agents, door, time_step, delay)


def update(frame):
    plt.clf()  # Clear the previous plot
    agent_data_animation, door_data_animation = ran_simulation
    # Plot agents
    agent_data_frame = agent_data_animation[agent_data_animation['Time'] == frame]
    door_data_frame = door_data_animation[door_data_animation['Time'] == frame]
    # Define a colormap based on competitiveness
    cmap = cm.get_cmap('viridis')  # You can change the colormap here
    competitiveness_values = agent_data_frame['Frustration']
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
    plt.colorbar(label='Frustration')  # Add a color bar

    # Add a marker as a train door
    plt.scatter(door_data_frame['Door X Position'], door_data_frame['Door Y Position'], marker='s', color='orange',
                s=300, label='Train Door')
    # plt.scatter(door_data_frame['Door X Position'], door_data_frame['Door Y Position'] - 0.8, marker='o', color='red',
    #             s=20, label='pole')
    # Add a marker as the stairs
    plt.scatter(stairs_location[0], stairs_location[1], marker='s', color='black', s=200, label='Stairs')

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
unique_times = range(1, num_timestamps*time_step,time_step)

# Create the animation
animation = FuncAnimation(fig, update, frames=unique_times, interval=50, repeat=True)
# To save the animation using Pillow as a gif
# writer = PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
# animation.save('boarding_animation.gif', writer=writer)
# Display the animation
plt.show()
