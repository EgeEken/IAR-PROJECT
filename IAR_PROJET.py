

# |=======================================================|
# |   PROJET IAR 2025/2026                                |
# |   ANDROIDE/AI2D - Sorbonne University                 |
# |=======================================================|
# |   Tarık Ege EKEN - 21110611                           |                              
# |   Kaan DISLI - 21113004                               |
# |=======================================================|
# |   Benoit GIRARD                                       |                             
# |=======================================================|
# |   A model of hippocampally dependent navigation,      |
# |   using the temporal difference learning rule         |
# |   Foster Morris Dayan. (2000)                         |
# |=======================================================|

import numpy as np
import matplotlib.pyplot as plt
import time
import random


class Environment:
    def __init__(self, radius=1.0, platform_radius=0.1, speed=0.3, momentum_weight_ratio=3, dt=0.1, timeout=120, timeout_penalty=0):
        """
        ## Simulation Environment
        ### Default values:
        - radius = **1.0 m** (Pool radius)
        - platform_radius = **0.1 m** (Platform radius)
        - speed = **0.3 m/s** (Swimming speed)
        - momentum_weight_ratio = **3** (Momentum to control ratio (1:3))
        - dt = **0.1 s** (Time step / seconds per step)
        - timeout = **120 s** (Max time per trial)

        ## Sources from the paper:
        ### Environment Description:
        *"We simulated the swimming behavior of a rat in a **2m** diameter circular watermaze,
        which contained a **0.1m** diameter escape platform."*
        
        ### Movement Description:
        *"The swimming speed of the rat was constant, at **0.3 m/s**."*

        *"The walls were treated as reflecting boundaries: the rat ‘‘bounced’’ off.
        Any move into the platform area was counted as a move onto the platform."*

        "*To model momentum, the direction the rat heads was given by a
        mixture of control as specified by the actor, and the previous
        heading, in the ratio **1:3**."*
        
        ### Discretization:
        *"Space was treated as a continuous variable; however, 
        time was discretized into steps of **0.1 s**."*

        ### Starting Locations:
        *"Following the experimental protocols, each trial began at one of
        four starting locations located at the **north, south, east, and west**
        edges of the pool, and ended when either the rat reached the
        platform, or a time-out of **120 s** was reached."*

        ### Source: Foster Morris Dayan. (2000)
        """

        # Environment parameters
        self.radius = radius
        self.platform_radius = platform_radius
        # Movement parameters
        self.speed = speed
        self.momentum_weight_ratio = momentum_weight_ratio
        # Time parameters
        self.dt = dt
        self.step_distance = self.speed * self.dt # Distance moved per step
        # Timeout
        self.timer = 0
        self.timeout = timeout
        self.timeout_penalty = timeout_penalty

        # Move history
        self.move_history = []
        
        # Starting locations (N, S, E, W edges)
        self.start_locations = [
            np.array([0.0, radius * 0.9]),  # North
            np.array([0.0, -radius * 0.9]), # South
            np.array([radius * 0.9, 0.0]),  # East
            np.array([-radius * 0.9, 0.0])  # West
        ]

        # Initialize agent position randomly
        self.pos = np.zeros(2) # Position [x, y]
        self.previous_move_vector = np.zeros(2) # Momentum (initially stationary so 0)
        self.platform_pos = np.zeros(2) # Platform position [x, y]
        self.reset("DMP") # DMP because we want a random platform position at start

    def set_platform_position(self, pos=None):
        """Sets the platform location. If None, picks a random spot."""
        if pos is not None:
            self.platform_pos = np.array(pos)
        else:
            # Pick random location avoiding edges
            r = np.random.uniform(0.2 * self.radius, 0.8 * self.radius)
            theta = np.random.uniform(0, 2 * np.pi)
            self.platform_pos = np.array([r * np.cos(theta), r * np.sin(theta)])

    def reset(self, mode="DMP", start_pos=None):
        """Resets the environment to a starting position."""
        
        # Set platform position based on mode
        if mode == "DMP":
            self.set_platform_position() # Random new position
        elif mode == "RMW":
            # Position remains in the same location throughout the simulation
            pass
        else:
            raise ValueError("Invalid mode. Choose 'DMP' or 'RMW'.")

        # Reset position to start location
        if start_pos is not None:
            if type(start_pos) != list and type(start_pos) != np.ndarray:
                self.pos = self.start_locations[start_pos].copy()
            else:
                self.pos = np.array(start_pos)
        else:
            self.pos = self.start_locations[np.random.randint(0, 4)].copy()
        
        # Reset momentum (rat starts stationary)
        self.previous_move_vector = np.zeros(2)

        # Reset timer
        self.timer = 0

        # Reset move history
        self.move_history = []

    def step(self, action_vector):
        """Takes a step in the environment based on the action vector."""
        
        # Update timer
        self.timer += self.dt
        if self.timer >= self.timeout:
            # Timeout reached
            return self.timeout_penalty, True
        
        # Update Move History
        self.move_history.append(self.pos.copy())

        # Normalize action vector
        action_norm = np.linalg.norm(action_vector)
        if action_norm > 0:
            action_vector = action_vector / action_norm
            
        intended_move = action_vector * self.step_distance
        
        # Momentum
        movement_vector = (intended_move + self.momentum_weight_ratio * self.previous_move_vector) / (1.0 + self.momentum_weight_ratio)
        
        # Normalize momentum vector
        movement_norm = np.linalg.norm(movement_vector)
        if movement_norm > 0:
            final_move = (movement_vector / movement_norm) * self.step_distance
        else:
            final_move = np.zeros(2)
        
        # Collision
        # Paper:
        #   "The walls were treated as reflecting boundaries: the rat "bounced" off."
        # A little ambiguous IMO but I assume it means something like this:

        # Movement vector collides with wall, the part that goes beyond the wall
        # is reflected back inside towards the center. Which kinda matches the sharp 
        # "bouncing" turns when the agent hits the wall that we see in the graphs in the paper.

        new_pos = self.pos + final_move
        dist_to_center = np.linalg.norm(new_pos)
        if dist_to_center > self.radius:
            # Calculate the overshoot distance
            overshoot = dist_to_center - self.radius
            # Calculate collision point on the boundary
            collision_point = (new_pos / dist_to_center) * self.radius # I think this is right? Hard to do the math in my head
            #print(f"collision_point: {collision_point}")
            # Reflect the overshoot back towards the center from the collision point to get new movement direction
            final_move = final_move - 2 * (np.dot(final_move, collision_point) / np.dot(collision_point, collision_point)) * collision_point
            # Normalize movement vector to get new final move after collision
            final_move = (final_move / np.linalg.norm(final_move)) * self.step_distance
            # New position after reflection using the final move direction, multiplied by the overshoot
            new_pos = collision_point + final_move * overshoot

        self.pos = new_pos
        self.previous_move_vector = final_move
        
        # 4. Check Reward (Platform)
        dist_to_goal = np.linalg.norm(self.pos - self.platform_pos)
        
        reward = 0
        done = False
        
        if dist_to_goal < self.platform_radius:
            reward = 1
            done = True
            
        return reward, done
    
    def display(self, show_move_history=True, plt_show=True, plt_close=False, show_vector=False, save_path=None, attempted_move=None, return_image=False, title=None):
        """Displays the current state of the environment.
        Green line shows the path taken by the agent if show_move_history is True.
        A black arrow shows the current movement vector if show_vector is True.
        If save_path is provided, saves the figure to that path.
        If return_image is True, returns the image as a numpy array instead of displaying it.
        """
        
        fig = plt.figure(figsize=(6, 6))
        circle = plt.Circle((0, 0), self.radius, color='blue', fill=False)
        platform = plt.Circle(self.platform_pos, self.platform_radius, color='green', alpha=0.7, label='Platform')
        plt.gca().add_artist(circle)
        plt.gca().add_artist(platform)
        plt.scatter(self.pos[0], self.pos[1], c='red', label='Agent')
        plt.xlim(-self.radius-0.1, self.radius+0.1)
        plt.ylim(-self.radius-0.1, self.radius+0.1)
        plt.gca().set_aspect('equal', adjustable='box')

        if title is not None:
            plt.title(title)

        if show_vector and np.linalg.norm(self.previous_move_vector) > 0:
            plt.arrow(self.pos[0], self.pos[1], self.previous_move_vector[0]*3, self.previous_move_vector[1]*3, 
                    color='black', alpha=0.5, head_width=0.02, length_includes_head=True, label="Movement Vector")

        if show_vector and attempted_move is not None:
            attempted_norm = np.linalg.norm(attempted_move)
            if attempted_norm > 0:
                attempted_move_normalized = (attempted_move / attempted_norm) * self.step_distance
                plt.arrow(self.pos[0], self.pos[1], attempted_move_normalized[0]*3, attempted_move_normalized[1]*3, 
                        color='orange', alpha=0.5, head_width=0.02, length_includes_head=True, label='Attempted Move')

        if show_move_history and len(self.move_history) > 1:
            move_history_array = np.array(self.move_history)
            plt.plot(move_history_array[:, 0], move_history_array[:, 1], marker=".", color='green', alpha=0.4, label='Move History')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if return_image:
            import io
            from PIL import Image
            # Save to buffer and read back as image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            image_array = np.array(img)
            plt.close(fig)
            buf.close()
            return image_array

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')

        if plt_show:
            plt.show()
        if plt_close:
            plt.close()


class DemoEnv:
    def __init__(self, radius=1.0, platform_radius=0.1, speed=0.3, momentum_weight_ratio=3, dt=0.1, timeout=120, timeout_penalty=0):
        self.env = Environment(radius=radius, platform_radius=platform_radius, speed=speed, momentum_weight_ratio=momentum_weight_ratio, dt=dt, timeout=timeout, timeout_penalty=timeout_penalty)

    def step_function_demo(self, max_steps=100, fps=10, start_pos=None, move_dir=None, save_path="animation.mp4", title="Environment Demo"):
        import cv2
        from IPython.display import Video

        if start_pos is None:
            start_pos = self.env.start_locations[np.random.randint(0, 4)].copy()
        self.env.reset("RMW", start_pos=start_pos)
        done = False
        step_count = 0
        temp_move_dir = move_dir

        img_array = []

        # Frame capture loop
        while not done and step_count < max_steps:
            step_count += 1
            if move_dir is None:
                temp_move_dir = np.random.uniform(-1, 1, size=2)
            _, done = self.env.step(np.array(temp_move_dir))
            img_rgb = self.env.display(show_move_history=False, show_vector=True, attempted_move=temp_move_dir, plt_show=False, return_image=True, title=title)
            if img_rgb.shape[2] == 4:
                img_rgb = img_rgb[:, :, :3]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_array.append(img_bgr)

        # Final display
        self.env.display(plt_show=True, title=title)
        
        # Create video
        if img_array:
            height, width = img_array[0].shape[:2]
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
            for img in img_array:
                out.write(img)
            out.release()


class PlaceCells:
    def __init__(self, env, n_cells=493, sigma=0.16):
        """Hippocampal Place Cells"""
        self.env = env
        self.n_cells = n_cells
        self.sigma = sigma
        self.centers = self.generate_centers(n_cells)
        
    def generate_centers(self, n):
        angles = np.random.uniform(0, 2 * np.pi, n)
        rs = self.env.radius * np.sqrt(np.random.uniform(0, 1, n))
        x = rs * np.cos(angles)
        y = rs * np.sin(angles)
        return np.array([[x[i], y[i]] for i in range(n)])
    
    def display_centers(self, highlight_id=None, plt_show=True):
        fig = plt.figure(figsize=(6, 6))
        circle = plt.Circle((0, 0), self.env.radius, color='blue', fill=False)
        plt.gca().add_artist(circle)
        #plt.scatter(self.centers[:, 0], self.centers[:, 1], s=self.sigma*1000, c='red', label='Place Cell Centers', alpha=0.6)
        # make each place cells edges more transparent than the centers (not of the environment, of each cell)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], s=500*self.sigma, c='red', label='Place Cell Centers', alpha=0.6, edgecolors='none')
        if highlight_id is not None and 0 <= highlight_id < self.n_cells:
            plt.scatter(self.centers[highlight_id, 0], self.centers[highlight_id, 1], s=600*self.sigma, c='yellow', label=f'Highlighted Cell ({highlight_id})', edgecolors='black')
        plt.xlim(-self.env.radius-0.1, self.env.radius+0.1)
        plt.ylim(-self.env.radius-0.1, self.env.radius+0.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Place Cell Centers")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if plt_show:
            plt.show()

    def get_activation(self, pos):
        """
        Activation Formula (Equation 1, Page 4): f_i(p) = exp( - (||p - s_i||^2) / (2 * sigma^2) )
        """
        # diff = || p - s_i ||
        diff = np.linalg.norm(pos - self.centers, axis=1)
        dist_sq = diff**2
        return np.exp(-dist_sq / (2 * self.sigma**2))
    
    def display_activation_field(self, place_cell_id, grid_size=100, plt_show=True):
        x = np.linspace(-self.env.radius, self.env.radius, grid_size)
        y = np.linspace(-self.env.radius, self.env.radius, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                activation = self.get_activation(pos)
                Z[i, j] = activation[place_cell_id]

        fig = plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        circle = plt.Circle((0, 0), self.env.radius, color='blue', fill=False)
        plt.gca().add_artist(circle)
        plt.colorbar(label='Activation Level')
        plt.title(f'Activation Field of Place Cell {place_cell_id}')
        plt.xlim(-self.env.radius-0.1, self.env.radius+0.1)
        plt.ylim(-self.env.radius-0.1, self.env.radius+0.1)
        plt.gca().set_aspect('equal', adjustable='box')
        if plt_show:
            plt.show()

    def display_all_place_activation_by_location(self, pos=None, plt_show=True, top_k=10):
        # Display a plot of the activation of all place cells when the agent is at position pos, as a scatter plot with color intensity representing activation level
        # Also a legend on the side showing activation levels for the top 30 activated place cells with their ids and activation values
        # Display these as a scatter on the environment, and the pos as a red dot
        if pos is None:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, self.env.radius)
            pos = np.array([r * np.cos(angle), r * np.sin(angle)])
        activation = self.get_activation(pos)
        fig = plt.figure(figsize=(13, 6))
        circle = plt.Circle((0, 0), self.env.radius, color='blue', fill=False)
        plt.gca().add_artist(circle)
        sc = plt.scatter(self.centers[:, 0], self.centers[:, 1], s=500*self.sigma, c=activation, cmap='viridis', alpha=0.8, edgecolors='black')
        plt.scatter(pos[0], pos[1], s=200, c='red', label='Agent Position', edgecolors='black')
        plt.xlim(-self.env.radius-0.1, self.env.radius+0.1)
        plt.ylim(-self.env.radius-0.1, self.env.radius+0.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Place Cell Activations at Position {pos[0]:.2f}, {pos[1]:.2f}')
        plt.colorbar(sc, label='Activation Level')
        # Top k activations
        top_k_indices = np.argsort(activation)[-top_k:][::-1]
        legend_text = "Top Place Cells:\n"
        for idx in top_k_indices:
            legend_text += f"ID {idx}: {activation[idx]:.2f}\n"
        plt.gcf().text(0.85, 0.5, legend_text, fontsize=10, va='center')
        if plt_show:
            plt.show()


class TD_Agent:
    def __init__(self, env, n_cells=493, sigma=0.16, learning_rate=0.02, discount_factor=0.99, lambda_param=0.8):
        """TD Agent (Actor-Critic)"""
        self.env = env
        self.alpha = learning_rate 
        self.gamma = discount_factor
        self.lambd = lambda_param
        self.n_cells = n_cells
        
        # Place Cells
        self.place_cells = PlaceCells(env, n_cells=n_cells, sigma=sigma)
        
        # Actor 
        self.actor = Actor(self, n_cells, n_actions=8)
        self.critic = Critic(self, n_cells)

    def get_value(self, activation):
        return self.critic.forward(activation)
    
    def get_action_probabilities(self, activation):
        return self.actor.get_action_probabilities(activation)
    
    def get_action_vectors(self):
        return self.actor.action_vectors
    
    def update(self, current_activation, action_idx, delta):
        # Traces
        self.critic.update_trace(current_activation, self.gamma, self.lambd)
        self.actor.update_trace(current_activation, action_idx, self.gamma, self.lambd)
        # Weights
        self.critic.update_weights(delta, self.alpha)
        self.actor.update_weights(delta, self.alpha)

    def run_trial(self, mode="DMP", max_steps=2000):
        self.env.reset(mode=mode)
        self.actor.reset_trace()
        self.critic.reset_trace()

        steps = 0
        path_length = 0
        done = False

        current_activation = self.place_cells.get_activation(self.env.pos)

        while not done and steps < max_steps:
            # Actor 
            action_vec, action_idx = self.actor.select_action(current_activation)

            # Move
            prev_pos = self.env.pos.copy()
            reward, done = self.env.step(action_vec)
            path_length += np.linalg.norm(self.env.pos - prev_pos)

            # TD Critic Error Calculation
            next_activation = self.place_cells.get_activation(self.env.pos)
            v_curr = self.critic.forward(current_activation)
            v_next = self.critic.forward(next_activation)

            delta = reward + self.gamma * v_next - v_curr

            # Update
            self.update(current_activation, action_idx, delta)

            current_activation = next_activation
            steps += 1

        trial_time = steps * self.env.dt
        return steps, path_length, trial_time, done
    
    def run_day(self, mode="RMW", trials_per_day=4):
        results = []
        # If DMP, platform moves at start of day, then stays for 4 trials
        if mode == "DMP":
            self.env.set_platform_position()
        
        for _ in range(trials_per_day):
            # Pass RMW here so env doesn't move the platform between trials
            # (If mode was DMP, we already set the pos above, so we treat trials as RMW relative to that pos)
            res = self.run_trial(mode="RMW") 
            results.append(res)
        return results
    
    def run_experiment(self, trials_by_day=["RMW"]*9, trials_per_day=4):
        all_results = []
        for day_mode in trials_by_day:
            day_results = self.run_day(mode=day_mode, trials_per_day=trials_per_day)
            all_results.append(day_results)
        return np.array(all_results)
    
    def run_figure4(self, simulation_count=1, trials_by_day=["RMW"]*9, trials_per_day=4):
        all_simulation_results = []
        for _ in range(simulation_count):
            sim_results = self.run_experiment(trials_by_day=trials_by_day, trials_per_day=trials_per_day)
            all_simulation_results.append(sim_results)
        return all_simulation_results

    def display_figure4(self, all_simulation_results, plt_show=True):
        data = np.array(all_simulation_results)
                
        simulation_count, day_count, trials_per_day, _ = data.shape
        path_lengths = data[:, :, :, 1]

        means = np.mean(path_lengths, axis=(0))
        stds = np.std(path_lengths, axis=(0))

        flat_means = means.flatten()
        flat_stds = stds.flatten()

        # x axis positions for space between days
        plt.figure(figsize=(12, 6))
        space_between_days = 3
        x_positions = []
        day_positions = []

        # Generate x_positions for each trial and group them by day
        for day in range(day_count):
            day_x_positions = []
            for trial in range(trials_per_day):
                pos = day * (trials_per_day + space_between_days) + trial + 1
                x_positions.append(pos)
                day_x_positions.append(pos)
            day_positions.append(day_x_positions)

        # Plot each day's data separately with error bars
        for day, day_x_positions in enumerate(day_positions):
            plt.plot(day_x_positions, flat_means[day * trials_per_day:(day + 1) * trials_per_day], 
                    marker='o', color="black")
            plt.errorbar(
                day_x_positions,
                flat_means[day * trials_per_day:(day + 1) * trials_per_day],
                yerr=flat_stds[day * trials_per_day:(day + 1) * trials_per_day],
                fmt='o',
                color="black",
                ecolor="gray",
                capsize=5,
                alpha=0.5
            )

        # Day ticks
        major_ticks = [(day * (trials_per_day + space_between_days)) + trials_per_day / 2 for day in range(day_count)]
        plt.xticks(major_ticks, [str(day + 1) for day in range(day_count)])
        plt.xlabel("Day")
        plt.ylabel("Path Length (m)")

        # Trial ticks
        plt.gca().set_xticks(x_positions, minor=True)

        plt.grid(visible=True, which='both', axis='x', linestyle='--', alpha=0.5)
        plt.title(f"Figure 4a: Actor-Critic Learning Curve\n({simulation_count} Simulations, {trials_per_day} Trials per day, {day_count} Days)")
        if plt_show:
            plt.show()

class Actor:
    # TODO fix/verify the implementation of the Actor class (Paper page 5)
    def __init__(self, agent, n_inputs, n_actions=8):
        self.agent = agent
        self.n_inputs = n_inputs

        # From the paper:
        # For convenience, the rat is allowed to move in one of 
        # eight possible directions at each time step (e.g., north, northeast, east)
        self.n_actions = n_actions
        self.weights = np.zeros((n_inputs, n_actions))

        angles = np.linspace(0, 2*np.pi, self.n_actions, endpoint=False)
        self.action_vectors = np.array([[np.cos(a), np.sin(a)] for a in angles])
        self.trace = np.zeros((n_inputs, n_actions))

    def forward(self, place_activation):
        return np.dot(place_activation, self.weights)

    def get_action_probabilities(self, place_activation):
        action_values = self.forward(place_activation)
        # Formula (Equation 9, Page 5)
        probabilities = np.exp(2*action_values) / np.sum(np.exp(2*action_values))
        return probabilities, action_values
    
    def select_action(self, place_activation):
        probs, _ = self.get_action_probabilities(place_activation)
        action_idx = np.random.choice(self.n_actions, p=probs)
        action_vec = self.action_vectors[action_idx]
        return action_vec, action_idx

    def update_trace(self, place_activation, action_idx, gamma, lambd):
        self.trace *= (gamma * lambd)
        self.trace[:, action_idx] += place_activation

    def update_weights(self, delta, learning_rate):
        self.weights += learning_rate * delta * self.trace

    def reset_trace(self):
        self.trace = np.zeros((self.n_inputs, self.n_actions))

    def display_policy(self, grid_size=20, plt_show=True):
        x = np.linspace(-self.agent.env.radius, self.agent.env.radius, grid_size)
        y = np.linspace(-self.agent.env.radius, self.agent.env.radius, grid_size)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                activation = self.agent.place_cells.get_activation(pos)
                probs, _ = self.get_action_probabilities(activation)
                best_action_idx = np.argmax(probs)
                best_action_vec = self.action_vectors[best_action_idx]
                U[i, j] = best_action_vec[0]
                V[i, j] = best_action_vec[1]

        plt.figure(figsize=(6, 6))
        plt.quiver(X, Y, U, V, color='blue', alpha=0.7)
        circle = plt.Circle((0, 0), self.agent.env.radius, color='blue', fill=False)
        plt.gca().add_artist(circle)
        plt.title('Actor Policy Vector Field')
        plt.scatter(self.agent.env.platform_pos[0], self.agent.env.platform_pos[1], s=200, c='green', label='Platform', alpha=0.5, edgecolors='black')
        plt.xlim(-self.agent.env.radius-0.1, self.agent.env.radius+0.1)
        plt.ylim(-self.agent.env.radius-0.1, self.agent.env.radius+0.1)
        plt.gca().set_aspect('equal', adjustable='box')
        if plt_show:
            plt.show()


class Critic:
    # TODO fix/verify the implementation of the Critic class (Paper page 5)
    def __init__(self, agent, n_inputs):
        self.agent = agent
        self.n_inputs = n_inputs
        self.weights = np.zeros(n_inputs)
        self.trace = np.zeros(n_inputs)
    
    def forward(self, place_activation):
        return np.dot(self.weights, place_activation)
    
    def update_trace(self, place_activation, gamma, lambd):
        self.trace = (gamma * lambd) * self.trace + place_activation

    def update_weights(self, delta, learning_rate):
        self.weights += learning_rate * delta * self.trace

    def reset_trace(self):
        self.trace = np.zeros(self.n_inputs)

    def display_value_function(self, grid_size=100, plt_show=True):
        x = np.linspace(-self.agent.env.radius, self.agent.env.radius, grid_size)
        y = np.linspace(-self.agent.env.radius, self.agent.env.radius, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                activation = self.agent.place_cells.get_activation(pos)
                Z[i, j] = self.forward(activation)

        fig = plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        circle = plt.Circle((0, 0), self.agent.env.radius, color='blue', fill=False)
        plt.gca().add_artist(circle)
        plt.colorbar(label='Value Function')
        plt.title('Critic Value Function')
        plt.xlim(-self.agent.env.radius-0.1, self.agent.env.radius+0.1)
        plt.ylim(-self.agent.env.radius-0.1, self.agent.env.radius+0.1)
        plt.gca().set_aspect('equal', adjustable='box')
        if plt_show:
            plt.show()

























