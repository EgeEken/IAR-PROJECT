

# |=======================================================|
# |   PROJET IAR 2025 - 2026                              |
# |   ANDROIDE/AI2D - Sorbonne University                 |
# |=======================================================|
# |   Tarik Ege EKEN - **21110611**                       |                              
# |   Kaan DISLI - **21113004**                           |
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


class Environment:
    def __init__(self, radius=1.0, platform_radius=0.05, speed=0.3, momentum_weight_ratio=3, dt=0.1, timeout=120, timeout_penalty=0):
        """
        ## Simulation Environment
        ### Default values:
        - radius = **1.0 m** (Pool radius)
        - platform_radius = **0.05 m** (Platform radius)
        - speed = **0.3 m/s** (Swimming speed)
        - momentum_weight_ratio = **3** (Momentum to control ratio (1:3))
        - dt = **0.1 s** (Time step / seconds per step)
        - timeout = **120 s** (Max time per trial)

        ## Sources from the paper:
        ### Environment Description:
        *"We simulated the swimming behavior of a rat in a **2m** diameter circular watermaze,
        which contained a **0.1m** diameter escape platform."*
        (Note: radius = diameter / 2)
        
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
            np.array([0.0, radius * 0.99]),  # North
            np.array([0.0, -radius * 0.99]), # South
            np.array([radius * 0.99, 0.0]),  # East
            np.array([-radius * 0.99, 0.0])  # West
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

    def random_movement_vector(self):
        """Generates a random movement vector of length step_distance."""
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)]) * self.step_distance

    def reset(self, mode="DMP", start_pos=None, platform_pos=None):
        """Resets the environment to a starting position."""
        
        # Set platform position based on mode
        if mode == "DMP":
            if platform_pos is not None:
                self.set_platform_position(platform_pos)
            else:
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
        collision_detected = False
        # Update timer
        self.timer += self.dt
        if self.timer >= self.timeout:
            # Timeout reached
            return self.timeout_penalty, True, False
        
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
            collision_detected = True
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
            new_pos = collision_point + (final_move / self.step_distance)  * overshoot

        self.pos = new_pos
        self.previous_move_vector = final_move
        
        # 4. Check Reward (Platform)
        dist_to_goal = np.linalg.norm(self.pos - self.platform_pos)
        
        reward = 0
        done = False
        
        if dist_to_goal < self.platform_radius:
            reward = 1
            done = True
            
        return reward, done, collision_detected
    
    def display(self, show_move_history=True, plt_show=True, plt_close=False, show_vector=False, save_path=None, attempted_move=None, return_image=False, title=None):
        """Displays the current state of the environment.
        Green line shows the path taken by the agent if show_move_history is True.
        A black arrow shows the current movement vector if show_vector is True.
        If save_path is provided, saves the figure to that path.
        If return_image is True, returns the image as a numpy array instead of displaying it.
        """
        
        if return_image:
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
    def __init__(self, radius=1.0, platform_radius=0.05, speed=0.3, momentum_weight_ratio=3, dt=0.1, timeout=120, timeout_penalty=0):
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
            _, done, _ = self.env.step(np.array(temp_move_dir))
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
        """
        # Hippocampal Place Cells

        ### Default values:
        - n_cells = **493** (Number of place cells)
        - sigma = **0.16 m** (Place field width)

        ## Sources from the paper:

        ### Cell Count:
        *"We consider an ensemble of place cells **(N = 493)** with place 
        fields distributed in an overlapping manner throughout the maze

        ### Place Field Width (Sigma): 
        *"...each with width **s = 0.16 m**"*

        ### Source: Foster Morris Dayan. (2000)
        """
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
    def __init__(self, env, n_cells=493, sigma=0.16, actor_lr=0.1,
                  critic_lr=0.01, gamma=0.99, actor_lambda=0.9,
                    critic_lambda=0.9, concussion_amnesia=False):
        """
        # TD Agent (Actor-Critic)

        ### Default values:
        - n_cells = **493** (Number of place cells)
        - sigma = **0.16 m** (Place field width)
        - actor_lr = **0.1** (Actor learning rate)
        - critic_lr = **0.01** (Critic learning rate)
        - gamma = **0.99** (Discount factor)
        - actor_lambda = **0.9** (Actor Eligibility trace decay rate)
        - critic_lambda = **0.9** (Critic Eligibility trace decay rate)

        ## Sources from the paper:

        ### Place Cells:
        *"We consider an ensemble of place cells **(N = 493)** with place
        fields distributed in an overlapping manner throughout the maze"*

        ### Place Field Width (Sigma):
        *"...each with width **s = 0.16 m**"*

        ### Learning Rate:
        The learning rate is never explicitly specified in the paper, just says this:\n
        *"Following standard reinforcement learning practice, we use a **fixed
        learning rate** to avoid slow learning"*
        Assuming a common value of **0.1** for the actor and **0.01** for the critic here.
        These values also produced similar results to the paper's figures.

        ### Discount Factor (Gamma):
        The discount factor is also never explicitly specified in the paper, just says this:\n
        *"gamma is a constant discounting factor, set such that **0 < gamma < 1**"*
        Assuming a common value of **0.99** here.

        ### Eligibility Trace Decay Rate (Lambda):
        *"Simulations confirmed this, and so we set lambda to **0.9**."*
        
        ### Source: Foster Morris Dayan. (2000)
        """
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        
        # Place Cells
        self.place_cells = PlaceCells(env, n_cells=n_cells, sigma=sigma)
        
        # Actor 
        self.actor = Actor(self, n_cells, n_actions=8)
        self.critic = Critic(self, n_cells)

        # Bonus (testing ways to reduce variance in simulations)
        self.concussion_amnesia = concussion_amnesia

    def get_value(self, activation):
        return self.critic.forward(activation)
    
    def get_action_probabilities(self, activation):
        return self.actor.get_action_probabilities(activation)
    
    def get_action_vectors(self):
        return self.actor.action_vectors
    
    def reset_model(self):
        self.actor.reset_weights()
        self.critic.reset_weights()
        self.actor.reset_trace()
        self.critic.reset_trace()
    
    def update(self, current_activation, next_activation, action_idx, reward, done, collision_detected=False):
        v_curr = self.critic.forward(current_activation)

        if done:
            v_next = 0.0
        else:
            v_next = self.critic.forward(next_activation)

        delta = reward + self.gamma * v_next - v_curr

        self.critic.update(delta, self.critic_lr, current_activation, self.gamma, self.critic_lambda)
        self.actor.update(delta, self.actor_lr, current_activation, action_idx, self.gamma, self.actor_lambda)

        if collision_detected:
            self.actor.reset_trace()
            self.critic.reset_trace()

    def run_trial(self, mode="DMP", max_steps=2000, learning=True, use_argmax=False):
        self.env.reset(mode=mode)

        self.actor.reset_trace()
        self.critic.reset_trace()

        steps = 0
        path_length = 0
        done = False

        current_activation = self.place_cells.get_activation(self.env.pos)

        while not done and steps < max_steps:
            # Actor 
            action_vec, action_idx = self.actor.select_action(current_activation, use_argmax=use_argmax)

            # Move
            prev_pos = self.env.pos.copy()
            reward, done, collision_detected = self.env.step(action_vec)
            path_length += np.linalg.norm(self.env.pos - prev_pos)

            # TD Critic Error Calculation
            next_activation = self.place_cells.get_activation(self.env.pos)

            # Update
            if learning:
                if self.concussion_amnesia:
                    self.update(current_activation, next_activation, action_idx, reward, done, collision_detected)
                else:
                    self.update(current_activation, next_activation, action_idx, reward, done, False)

            current_activation = next_activation
            steps += 1

        trial_time = steps * self.env.dt
        return steps, path_length, trial_time, done
    
    def run_day(self, mode="RMW", trials_per_day=4, use_argmax=False):
        results = []
        # If DMP, platform moves at start of day, then stays for 4 trials
        if mode == "DMP":
            self.env.set_platform_position()
        
        for _ in range(trials_per_day):
            # Pass RMW here so env doesn't move the platform between trials
            # (If mode was DMP, we already set the pos above, so we treat trials as RMW relative to that pos)
            res = self.run_trial(mode="RMW", use_argmax=use_argmax)
            results.append(res)
        return results
    
    def run_experiment(self, trials_by_day=["RMW"]*9, trials_per_day=4, use_argmax=False):
        # Reset agent (weights & traces) before experiment
        self.reset_model()
        all_results = []
        for day_mode in trials_by_day:
            day_results = self.run_day(mode=day_mode, trials_per_day=trials_per_day, use_argmax=use_argmax)
            all_results.append(day_results)
        return np.array(all_results)
    
    def run_figure4(self, simulation_count=1, trials_by_day=["RMW"]*9, trials_per_day=4, use_argmax=False):
        all_simulation_results = []
        for _ in range(simulation_count):
            sim_results = self.run_experiment(trials_by_day=trials_by_day, trials_per_day=trials_per_day, use_argmax=use_argmax)
            all_simulation_results.append(sim_results)
        return all_simulation_results

    def display_figure4(self, all_simulation_results, plt_show=True, title=None):
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
            # I spent days trying to figure out why the error bars looked different than the paper's figure
            # even though i implemented the TD Agent exactly as described in the paper.
            # Finally figured it out, it's because of the error bars calculation, not the agent itself.
            # The paper uses "Standard Error of the Mean" for error bars
            # SEotM = std / sqrt(N)
            # "For each data point, the mean and standard error in the mean
            # are obtained from 1,000 simulation runs."
            # (Page 7, Fİgure 4 Caption)
            SEotM = flat_stds[day * trials_per_day:(day+1) * trials_per_day] / np.sqrt(simulation_count)
            plt.errorbar(
                day_x_positions,
                flat_means[day * trials_per_day:(day + 1) * trials_per_day],
                yerr=SEotM,
                fmt='o',
                color="black",
                ecolor="gray",
                capsize=5,
                alpha=0.5
            )

        # Day ticks
        major_ticks = [(day * (trials_per_day + space_between_days)) + trials_per_day / 2 for day in range(day_count)]
        plt.xticks(major_ticks, [str(day + 1) for day in range(day_count)])
        plt.yticks([5*i for i in range(0, 8)], [str(5*i) for i in range(0, 8)])
        plt.xlabel("Day")
        plt.ylabel("Path Length (m)")
        plt.ylim(0, 36)

        # Trial ticks
        ax = plt.gca()
        ax.set_xticks(x_positions, minor=True)
        ax.tick_params(axis="x", which="both", direction="in")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #plt.grid(visible=True, which='both', axis='x', linestyle='--', alpha=0.5)
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"Figure 4: Actor-Critic Learning Curve\n({simulation_count} Simulations, {trials_per_day} Trials per day, {day_count} Days)")
        if plt_show:
            plt.show()

class Actor:
    def __init__(self, agent, n_inputs, n_actions=8):
        """
        # Actor Component

        ### Default values:
        - n_actions = **8** (Number of possible movement directions)

        ## Sources from the paper:
        ### Action Directions:
        *"For convenience, the rat is allowed to move in one of
        **eight** possible directions at each time step (e.g., north, northeast, east)"*

        
        ### Source: Foster Morris Dayan. (2000)
        """
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

    def get_action_probabilities(self, place_activation, use_argmax=False):
        # Paper gives the temprature value 2 in the formulas
        action_values = self.forward(place_activation)
        # Not 100% sure if we need to prevent overflow here, paper says nothing about it but i get the error sometimes when running
        max_av = np.max(action_values)
        action_values -= max_av  # prevents overflow error
        # Formula (Equation 9, Page 5)
        e_2av = np.exp(2*action_values)
        probabilities = e_2av / np.sum(e_2av)
        if use_argmax:
            probabilities = np.zeros_like(probabilities)
            best_idx = np.argmax(action_values)
            probabilities[best_idx] = 1.0
        return probabilities, action_values
    
    def select_action(self, place_activation, use_argmax=False):
        probs, _ = self.get_action_probabilities(place_activation, use_argmax=use_argmax)
        action_idx = np.random.choice(self.n_actions, p=probs)
        action_vec = self.action_vectors[action_idx]
        return action_vec, action_idx
    
    def reset_weights(self):
        self.weights = np.zeros((self.n_inputs, self.n_actions))

    def reset_trace(self):
        self.trace = np.zeros((self.n_inputs, self.n_actions))

    def update(self, delta, alpha, place_activation, action_idx, gamma, lambd):
        # trace
        self.trace *= (gamma * lambd)
        self.trace[:, action_idx] += place_activation
        # clip trace (testing)
        self.trace = np.clip(self.trace, -10, 10)
        # weights
        self.weights += alpha * delta * self.trace

    def display_policy(self, grid_size=20, plt_show=True, title=None):
        r = self.agent.env.radius
        x = np.linspace(-r, r, grid_size)
        y = np.linspace(-r, r, grid_size)
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

        #plt.figure(figsize=(6, 6))
        plt.quiver(X, Y, U, V, color='blue', alpha=0.7)
        circle = plt.Circle((0, 0), r, color='blue', fill=False)
        plt.gca().add_artist(circle)

        platform_pos = self.agent.env.platform_pos
        platform = plt.Circle(platform_pos, self.agent.env.platform_radius, facecolor='green', alpha=0.7, label='Platform', edgecolor='black')
        plt.gca().add_artist(platform)

        plt.xlim(-r-0.1, r+0.1)
        plt.ylim(-r-0.1, r+0.1)
        if title is not None:
            plt.title(title)
        else:
            plt.title('Actor Policy Vector Field')
        plt.gca().set_aspect('equal', adjustable='box')
        if plt_show:
            plt.show()


class Critic:
    def __init__(self, agent, n_inputs):
        """
        # Critic Component
        """
        self.agent = agent
        self.n_inputs = n_inputs
        self.weights = np.zeros(n_inputs)
        self.trace = np.zeros(n_inputs)
    
    def forward(self, place_activation):
        return np.dot(self.weights, place_activation)
    
    def reset_weights(self):
        self.weights = np.zeros(self.n_inputs)

    def reset_trace(self):
        self.trace = np.zeros(self.n_inputs)

    def update(self, delta, alpha, place_activation, gamma, lambd):
        # trace
        self.trace = gamma * lambd * self.trace + place_activation
        # clip trace (testing)
        self.trace = np.clip(self.trace, -10, 10)
        # weights
        self.weights += alpha * delta * self.trace

    def display_value_function(self, grid_size=100, plt_show=True):
        r = self.agent.env.radius
        x = np.linspace(-r, r, grid_size)
        y = np.linspace(-r, r, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                activation = self.agent.place_cells.get_activation(pos)
                Z[i, j] = self.forward(activation)

        plt.contourf(X, Y, Z, levels=50, cmap='viridis')

        circle = plt.Circle((0, 0), r, color='blue', fill=False)
        plt.gca().add_artist(circle)

        platform_pos = self.agent.env.platform_pos
        platform = plt.Circle(platform_pos, self.agent.env.platform_radius, facecolor='green', alpha=0.7, label='Platform', edgecolor='black')
        plt.gca().add_artist(platform)
        
        plt.colorbar(label='Value Function')
        plt.title('Critic Value Function')
        plt.xlim(-r-0.1, r+0.1)
        plt.ylim(-r-0.1, r+0.1)
        # limit z axis between 0 and 1?
        plt.clim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        if plt_show:
            plt.show()

class Coordinates:
    def __init__(self, agent, n_cells=493, learning_rate=0.01, lambd=0.9):
        """
        # Coordinate Estimation System
        Paper Pages 8-11.

        ### Default values:
        - n_cells = **493** (Number of place cells)
        - learning_rate = **0.01** (Coordinate learning rate)
        - lambda = **0.9** (Eligibility trace decay rate)

        ## Sources from the paper:
        ### Cell Count:
        No specific mention of changing the number of place cells for the coordinate system,
        so we use the same as the previous Place Cell count: **493**.

        ### Learning Rate:
        The learning rate is never explicitly specified in the paper, just says this:\n
        *"Following standard reinforcement learning practice, we use a **fixed
        learning rate** to avoid slow learning"*\n
        Assuming a common value of **0.01** here.

        ### Lambda:
        *"Theoretical arguments suggest that since the terms 
        Dxt and Dyt are likely to be quite accurate, distant timesteps are
        useful, and therefore a high value of lambda should make learning fastest (Watkins, 1989).*\n
        *Simulations confirmed this, and so we set lambda to **0.9**."*

        """
        self.agent = agent
        self.n_cells = n_cells
        self.lr = learning_rate
        self.lambd = lambd
        
        self.weights_x = np.zeros(n_cells)
        self.weights_y = np.zeros(n_cells)
        
        self.trace_x = np.zeros(n_cells)
        self.trace_y = np.zeros(n_cells)

        # Goal memory will be updated to a 2D Coordinate when set_goal is called
        self.goal_memory = None

    def get_coordinates(self, place_activation):
        x = np.dot(self.weights_x, place_activation)
        y = np.dot(self.weights_y, place_activation)
        return np.array([x, y])

    def reset_traces(self):
        self.trace_x = np.zeros(self.n_cells)
        self.trace_y = np.zeros(self.n_cells)

    def update(self, current_activation, next_activation, real_movement_vector):
        # Update Traces
        self.trace_x = self.lambd * self.trace_x + current_activation
        self.trace_y = self.lambd * self.trace_y + current_activation

        # Estimates of current and next positions
        curr_coords = self.get_coordinates(current_activation) # [x, y]
        next_coords = self.get_coordinates(next_activation)    # [x, y]

        # Actual self-motion (Ground Truth)
        dx_real = real_movement_vector[0]
        dy_real = real_movement_vector[1]

        self.trace_x = np.clip(self.trace_x, -10.0, 10.0)
        self.trace_y = np.clip(self.trace_y, -10.0, 10.0)

        delta_x = -dx_real + next_coords[0] - curr_coords[0]
        delta_y = -dy_real + next_coords[1] - curr_coords[1]

        # Update Weights
        self.weights_x += self.lr * delta_x * self.trace_x
        self.weights_y += self.lr * delta_y * self.trace_y

    def set_goal(self, place_activation):
        self.goal_memory = self.get_coordinates(place_activation)

    def get_vector_to_goal(self, place_activation):
        if self.goal_memory is None:
            # "When there is no goal coordinate in memory... specify random, exploratory actions."
            # (Page 9, "Using Coordinates to Control Actions" section)
            rnd = np.random.uniform(-1, 1, 2)
            return rnd / np.linalg.norm(rnd), False
        
        curr_coords = self.get_coordinates(place_activation)
        diff = self.goal_memory - curr_coords
        dist = np.linalg.norm(diff)
        
        if dist > 0:
            return diff / dist, True
        else:
            return np.zeros(2), True
        
    def coordinate_center(self, cell_axis="x"):
        activation = self.agent.place_cells.get_activation(np.array([0, 0]))
        if cell_axis == "x":
            return np.dot(self.weights_x, activation)
        else:
            return np.dot(self.weights_y, activation)
    
    def coordinate_error(self, cell_axis="x", grid_size=100):
        c_center = self.coordinate_center(cell_axis=cell_axis)
        c_error = 0.0
        r = self.agent.env.radius
        x = np.linspace(-r, r, grid_size)
        y = np.linspace(-r, r, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                activation = self.agent.place_cells.get_activation(pos)
                if cell_axis == 'x':
                    Z[i, j] = np.dot(self.weights_x, activation) - c_center
                    c_error += abs(Z[i, j] - pos[0])
                else:
                    Z[i, j] = np.dot(self.weights_y, activation) - c_center
                    c_error += abs(Z[i, j] - pos[1])

        return c_error / (grid_size * grid_size)


    def get_activation_map(self, cell_axis, grid_size=100):
        r = self.agent.env.radius
        x = np.linspace(-r, r, grid_size)
        y = np.linspace(-r, r, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                activation = self.agent.place_cells.get_activation(pos)
                if cell_axis == 'x':
                    Z[i, j] = np.dot(self.weights_x, activation)
                else:
                    Z[i, j] = np.dot(self.weights_y, activation)

        return Z
        
    def display_cell_activity(self, cell_axis, grid_size=100, plt_show=True, show_perceived_target=False):
        r = self.agent.env.radius
        x = np.linspace(-r, r, grid_size)
        y = np.linspace(-r, r, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = self.get_activation_map(cell_axis, grid_size=grid_size)

        plt.contourf(X, Y, Z, levels=50, cmap='viridis')

        circle = plt.Circle((0, 0), r, color='blue', fill=False)
        plt.gca().add_artist(circle)

        platform_pos = self.agent.env.platform_pos
        platform = plt.Circle(platform_pos, self.agent.env.platform_radius, facecolor='green', alpha=0.7, label='Platform', edgecolor='black')
        plt.gca().add_artist(platform)

        if show_perceived_target and self.goal_memory is not None:
            perceived_target = plt.Circle(self.goal_memory, 0.05, facecolor='red', alpha=0.6, label='Perceived Target', edgecolor='black')
            plt.gca().add_artist(perceived_target)
        
        plt.colorbar(label=f'Coordinate Cell ({cell_axis.upper()}) Activity')
        plt.title(f'Coordinate Cell ({cell_axis.upper()}) Activity Map')
        plt.xlim(-r-0.1, r+0.1)
        plt.ylim(-r-0.1, r+0.1)
        plt.clim(-1, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        if plt_show:
            plt.show()

    def display_ideal_cell_activity(self, cell_axis, grid_size=100, plt_show=True):
        r = self.agent.env.radius
        x = np.linspace(-r, r, grid_size)
        y = np.linspace(-r, r, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                if cell_axis == 'x':
                    Z[i, j] = pos[0]
                else:
                    Z[i, j] = pos[1]

        plt.contourf(X, Y, Z, levels=50, cmap='viridis')

        circle = plt.Circle((0, 0), r, color='blue', fill=False)
        plt.gca().add_artist(circle)

        platform_pos = self.agent.env.platform_pos
        platform = plt.Circle(platform_pos, self.agent.env.platform_radius, facecolor='green', alpha=0.7, label='Platform', edgecolor='black')
        plt.gca().add_artist(platform)
        
        plt.colorbar(label=f'Ideal Coordinate ({cell_axis.upper()}) Value')
        plt.title(f'Ideal Coordinate ({cell_axis.upper()}) Map')
        plt.xlim(-r-0.1, r+0.1)
        plt.ylim(-r-0.1, r+0.1)
        plt.clim(-1, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        if plt_show:
            plt.show()




class Coordinate_TD_Agent:
    def __init__(self, env, n_cells=493, sigma=0.16, 
                 actor_lr=0.1, critic_lr=0.01, coord_lr=0.01,
                 gamma=0.99, lambd=0.9, concussion_amnesia=False,
                 goal_memory_reset_interval=20, coord_action_scale1=1.0, coord_action_scale2=1.0,
                 coord_action_lr=0.1, panic_radius=1.0
                 ):
        """
        # Coordinate Based Navigation Agent
        Combines the Actor-Critic architecture with a learned Coordinate System.
        Paper Pages 8-11.

        ### Default values:
        - n_cells = **493** (Number of place cells)
        - sigma = **0.16 m** (Place field width)
        - actor_lr = **0.1** (Actor learning rate)
        - critic_lr = **0.01** (Critic learning rate)
        - coord_lr = **0.01** (Coordinate system learning rate)
        - gamma = **0.99** (Discount factor)
        - lambda = **0.9** (Eligibility trace decay rate for Actor, Critic, and Coordinates)

        ## Sources from the paper:

        ### The Coordinate System:
        *"The coordinate system consists of... a coordinate representation of current position
        (X and Y)... a goal coordinate memory... and a mechanism which computes the direction
        to swim."*

        ### The Abstract Action (9th Action):
        *"Here, there is an additional action cell, a_coord, representing the rat’s preference
        for the swimming direction offered by the coordinate system... The coordinate action
        is reinforced by the critic in a similar manner to the other actions."*

        ### Lambda:
        *"Theoretical arguments suggest that... distant timesteps are useful, and therefore
        a high value of lambda should make learning fastest... Simulations confirmed this,
        and so we set lambda to **0.9**."*

        ### Goal Memory:
        *"When there is no remembered goal coordinate... the controller specifies random,
        exploratory actions... then the controller does not participate in learning,
        i.e., a_coord is not updated."*
        
        ### Source: Foster Morris Dayan. (2000)
        """
        
        self.env = env
        self.place_cells = PlaceCells(env, n_cells=n_cells, sigma=sigma)
        
        # Hyperparameters
        self.actor_lr = actor_lr
        self.coord_action_lr = coord_action_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lambd = lambd
        
        # Critic
        self.critic = Critic(self, n_cells)
        
        # Coordinate System
        self.coordinates = Coordinates(self, n_cells, learning_rate=coord_lr, lambd=lambd)
        
        # Actor
        # We need 9 output units: 0-7 are fixed directions, 8 is the "Coordinate Action"
        # We need to fix the first 8 weights separately to allow for the 9th action to be learned differently
        self.n_actions_total = 9
        self.actor_weights_fixed = np.zeros((n_cells, 8))
        self.actor_trace_fixed = np.zeros((n_cells, 8))

        # Scalar weights and trace for the coordinate action
        self.coord_action_weight = 0.0
        self.coord_action_trace = 0.0
        
        # Fixed vectors for actions 0-7
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        self.fixed_action_vectors = np.array([[np.cos(a), np.sin(a)] for a in angles])

        # Bonus (testing)
        self.concussion_amnesia = concussion_amnesia
        self.goal_memory_reset_interval = goal_memory_reset_interval
        self.goal_memory_reset_timer = self.goal_memory_reset_interval
        self.coord_action_scale1 = coord_action_scale1
        self.coord_action_scale2 = coord_action_scale2
        self.panic_radius = panic_radius

    def reset_model(self):
        self.actor_weights_fixed = np.zeros_like(self.actor_weights_fixed)
        self.actor_trace_fixed = np.zeros_like(self.actor_trace_fixed)
        self.coord_action_weight = 0.0
        self.coord_action_trace = 0.0
        self.critic.reset_weights()
        self.critic.reset_trace()
        self.coordinates.weights_x = np.zeros(self.place_cells.n_cells)
        self.coordinates.weights_y = np.zeros(self.place_cells.n_cells)
        self.coordinates.reset_traces()
        self.coordinates.goal_memory = None

    def actor_forward(self, place_activation):
        return np.dot(place_activation, self.actor_weights_fixed)

    def get_action_probabilities(self, place_activation):
        # Calculate values for all 9 actions
        fixed_values = self.actor_forward(place_activation)
        coord_value = np.array([self.coord_action_weight * self.coord_action_scale1])
        action_values = np.concatenate((fixed_values, coord_value))
        max_av = np.max(action_values)
        action_values -= max_av
        e_2av = np.exp(2 * action_values)
        probs = e_2av / np.sum(e_2av)
        return probs
    
    def select_action(self, place_activation):
        probs = self.get_action_probabilities(place_activation)
        action_idx = np.random.choice(self.n_actions_total, p=probs)
        if action_idx < 8:
            action_vec = self.fixed_action_vectors[action_idx]
        else:
            action_vec, _ = self.coordinates.get_vector_to_goal(place_activation)
        return action_vec, action_idx, probs[8]

    def display_actor_policy(self, grid_size=20, plt_show=True, title=None):
        """unlike previous display_policy, this one handles the 9th coordinate action, which
        depends on the coordinate system, and isn't bound by the 8 fixed directions.
        """
        r = self.env.radius
        x = np.linspace(-r, r, grid_size)
        y = np.linspace(-r, r, grid_size)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                activation = self.place_cells.get_activation(pos)
                probs = self.get_action_probabilities(activation)
                best_action_idx = np.argmax(probs)
                
                if best_action_idx < 8:
                    best_action_vec = self.fixed_action_vectors[best_action_idx]
                else:
                    coord_vec, _ = self.coordinates.get_vector_to_goal(activation)
                    best_action_vec = coord_vec

                U[i, j] = best_action_vec[0]
                V[i, j] = best_action_vec[1]

        plt.quiver(X, Y, U, V, color='blue', alpha=0.7)
        circle = plt.Circle((0, 0), r, color='blue', fill=False)
        plt.gca().add_artist(circle)

        platform_pos = self.env.platform_pos
        platform = plt.Circle(platform_pos, self.env.platform_radius, facecolor='green', alpha=0.7, label='Platform', edgecolor='black')
        plt.gca().add_artist(platform)

        plt.xlim(-r-0.1, r+0.1)
        plt.ylim(-r-0.1, r+0.1)
        if title is not None:
            plt.title(title)
        else:
            plt.title('Coordinate TD Agent Actor Policy Vector Field')
        plt.gca().set_aspect('equal', adjustable='box')
        if plt_show:
            plt.show()


    def run_trial(self, mode="DMP", max_steps=2000, learning=True):
        self.actor_trace_fixed = np.zeros_like(self.actor_trace_fixed)
        self.coord_action_trace = 0.0
        self.critic.reset_trace()
        self.coordinates.reset_traces()
        self.env.reset(mode=mode)
        
        steps = 0
        path_length = 0
        done = False

        p_coordinate_action = 0
        
        self.goal_memory_reset_timer = self.goal_memory_reset_interval

        curr_act = self.place_cells.get_activation(self.env.pos)
        
        while not done and steps < max_steps:
            move_vec, action_idx, p_coordinate_action_step = self.select_action(curr_act)
            p_coordinate_action += p_coordinate_action_step

            if self.coordinates.goal_memory is not None:
                curr_coords = self.coordinates.get_coordinates(curr_act)
                goal_coords = self.coordinates.goal_memory
                dist_to_goal = np.linalg.norm(goal_coords - curr_coords)
                if dist_to_goal <= self.env.platform_radius * self.panic_radius:
                    #if self.goal_memory_reset_timer == self.goal_memory_reset_interval:
                        #print(f"Reached memorized goal but not done, starting goal memory reset countdown.")
                    self.goal_memory_reset_timer -= 1
                    #print(f"Goal memory reset timer: {self.goal_memory_reset_timer}")
                    if self.goal_memory_reset_timer <= 0:
                        #print(f"Goal memory reset timer elapsed, clearing goal memory.")
                        #self.env.display()
                        # We're at the goal but not done, so clearly the platform must have changed position
                        self.coordinates.goal_memory = None
                        # Random exploratory action from old platform position
                        action_idx = np.random.randint(0, 8)
                        move_vec = self.fixed_action_vectors[action_idx]

            prev_pos = self.env.pos.copy()
            reward, done, collision = self.env.step(move_vec)
            
            real_movement = self.env.pos - prev_pos
            path_length += np.linalg.norm(real_movement)
            
            next_act = self.place_cells.get_activation(self.env.pos)
            
            if learning:
                # Coordinate System Update
                self.coordinates.update(curr_act, next_act, real_movement)

                # TD Critic Update
                v_curr = self.critic.forward(curr_act)
                v_next = 0.0 if done else self.critic.forward(next_act)
                delta = reward + self.gamma * v_next - v_curr
                
                # Critic Update
                self.critic.update(delta, self.critic_lr, curr_act, self.gamma, self.lambd)
                
                # Actor Update
                if collision and self.concussion_amnesia:
                    self.actor_trace_fixed = np.zeros_like(self.actor_trace_fixed)
                    self.coord_action_trace = 0.0
                    self.critic.reset_trace()
                else:
                    self.actor_trace_fixed *= (self.gamma * self.lambd)
                    self.coord_action_trace *= (self.gamma * self.lambd)
                
                # "When there is no remembered goal coordinate... a_coord is not updated."
                if action_idx < 8:
                    self.actor_trace_fixed[:, action_idx] += curr_act
                elif self.coordinates.goal_memory is not None:
                    #self.coord_action_trace += 1.0
                    #print(np.sum(curr_act)) # for debugging, gives values around 15-30 which is too big and breaks the learning
                    #self.coord_action_trace += np.sum(curr_act)
                    #print(np.mean(curr_act))  # for debugging, gives values around 0.03-0.06 which is too small and breaks learning
                    #self.coord_action_trace += np.mean(curr_act)
                    self.coord_action_trace += self.coord_action_scale2

                #self.actor_trace_fixed = np.clip(self.actor_trace_fixed, -10.0, 10.0)
                #self.coord_action_trace = np.clip(self.coord_action_trace, -10.0, 10.0)

                self.actor_weights_fixed += self.actor_lr * delta * self.actor_trace_fixed

                if self.coordinates.goal_memory is not None:
                    self.coord_action_weight += self.coord_action_lr * delta * self.coord_action_trace

            curr_act = next_act
            steps += 1
            
        if done and reward > 0:
            # if platform was found, set goal to it
            platform_act = self.place_cells.get_activation(self.env.pos)
            self.coordinates.set_goal(platform_act)
            #print(f"Found platform at position {self.env.pos} in {steps} steps, setting goal memory.")
            
        return steps, path_length, steps*self.env.dt, done, p_coordinate_action / steps
    
    
    def run_day(self, mode="RMW", trials_per_day=4):
        results = []
        # If DMP, platform moves at start of day, then stays for 4 trials
        if mode == "DMP":
            self.env.set_platform_position()
        
        for _ in range(trials_per_day):
            # Pass RMW here so env doesn't move the platform between trials
            # (If mode was DMP, we already set the pos above, so we treat trials as RMW relative to that pos)
            res = self.run_trial(mode="RMW")[:-1]
            results.append(res)
        return results
    
    def run_experiment(self, trials_by_day=["RMW"]*9, trials_per_day=4):
        # Reset agent (weights & traces) before experiment
        self.reset_model()
        all_results = []
        for day_mode in trials_by_day:
            day_results = self.run_day(mode=day_mode, trials_per_day=trials_per_day)
            all_results.append(day_results)
        return np.array(all_results)
    
    def run_figure8(self, simulation_count=1, trials_by_day=["RMW"]*9, trials_per_day=4, verbose=False):
        all_simulation_results = []
        timer = time.time()
        for i in range(simulation_count):
            if verbose and simulation_count > 1:
                if simulation_count < 6 or (i % (simulation_count // 5) == 0):
                    print(f"Running simulation {i+1}/{simulation_count}...", end=' | ')
            sim_results = self.run_experiment(trials_by_day=trials_by_day, trials_per_day=trials_per_day)
            all_simulation_results.append(sim_results)
            if verbose and simulation_count > 1:
                if simulation_count < 6 or (i % (simulation_count // 5) == 0):
                    print(f"Time so far: {time.time() - timer:.2f} seconds")
        return all_simulation_results

    def display_figure8(self, all_simulation_results, plt_show=True, title=None):
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
            # The paper uses "Standard Error of the Mean" for error bars
            # SEotM = std / sqrt(N)
            # "For each data point, the mean and standard error in the mean
            # are obtained from 1,000 simulation runs."
            # (Page 7, Fİgure 4 Caption)
            plt.errorbar(
                day_x_positions,
                flat_means[day * trials_per_day:(day + 1) * trials_per_day],
                yerr=flat_stds[day * trials_per_day:(day + 1) * trials_per_day] / np.sqrt(simulation_count),
                fmt='o',
                color="black",
                ecolor="gray",
                capsize=5,
                alpha=0.5
            )

        # Day ticks
        major_ticks = [(day * (trials_per_day + space_between_days)) + trials_per_day / 2 for day in range(day_count)]
        plt.xticks(major_ticks, [str(day + 1) for day in range(day_count)])
        plt.yticks([5*i for i in range(0, 8)], [str(5*i) for i in range(0, 8)])
        plt.xlabel("Day")
        plt.ylabel("Path Length (m)")
        plt.ylim(0, 36)

        # Trial ticks
        ax = plt.gca()
        ax.set_xticks(x_positions, minor=True)
        ax.tick_params(axis="x", which="both", direction="in")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #plt.grid(visible=True, which='both', axis='x', linestyle='--', alpha=0.5)
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"Figure 8: Coordinate System + Actor-Critic Learning Curve\n({simulation_count} Simulations, {trials_per_day} Trials per day, {day_count} Days)")
        if plt_show:
            plt.show()