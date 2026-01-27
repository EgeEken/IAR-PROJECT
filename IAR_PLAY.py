import pygame
import numpy as np
import math
import sys
import random
import matplotlib.pyplot as plt
from datetime import datetime

# |=======================================================|
# |   IAR_PLAY.py - Human Playable Watermaze              |
# |   Based on Foster Morris Dayan (2000)                 |
# |=======================================================|

# ==============================================================================
# 1. SIMULATION PARAMETERS (Adjust these as requested)
# ==============================================================================

# Display Settings
SCREEN_SIZE = 800             # Window size in pixels (square)
PIXELS_PER_METER = 250        # Zoom level (Screen pixels per real meter)
FPS = 60

# Physical Dimensions (Meters)
POOL_RADIUS_M = 1.0           # Radius of the pool in meters
PLATFORM_RADIUS_M = 0.05      # Radius of the platform in meters
RAT_SPEED_M_S = 0.3           # Max swimming speed (m/s)

# Physics & Control
MOMENTUM_WEIGHT_RATIO = 15.0   # 3:1 ratio (Previous : New input)
DT = 2.0 / FPS                # Time step
TIMEOUT_SECONDS = 120         # Max time per trial

# Task Structure
TRIALS_PER_DAY = 4
RMW_DAYS_PHASE_1 = 3          # Days 1-7 (Fixed location - RMW style)
RMW_DAYS_PHASE_2 = 1          # Day 8 (New location - DMP style)
RMW_DAYS_PHASE_3 = 1          # Day 9 (Stay in new location - RMW style)

DMP_TOTAL_DAYS = 5            # Total days for DMP task


# Colors
COLOR_BG = (30, 30, 30)       # Dark Grey Background
COLOR_POOL = (50, 100, 200)   # Water Blue
COLOR_PLATFORM = (0, 255, 0)  # Green
COLOR_RAT = (255, 50, 50)     # Red
COLOR_TEXT = (255, 255, 255)  # White
COLOR_HUD_BG = (0, 0, 0)      # Black

# ==============================================================================
# 2. PHYSICS ENGINE & ENVIRONMENT
# ==============================================================================

class WaterMazePhysics:
    def __init__(self):
        self.pos = np.zeros(2)      # x, y in meters
        self.velocity = np.zeros(2) # dx, dy vector
        self.platform_pos = np.zeros(2)
        self.start_locations = [
            np.array([0.0, POOL_RADIUS_M * 0.95]),  # North
            np.array([0.0, -POOL_RADIUS_M * 0.95]), # South
            np.array([POOL_RADIUS_M * 0.95, 0.0]),  # East
            np.array([-POOL_RADIUS_M * 0.95, 0.0])  # West
        ]
        self.timer = 0.0
        self.finished = False
        self.success = False

    def reset_trial(self, fixed_platform_pos=None):
        # 1. Pick Start Position (Random N, S, E, W)
        start_idx = random.randint(0, 3)
        self.pos = self.start_locations[start_idx].copy()
        
        # 2. Reset Physics
        self.velocity = np.zeros(2)
        self.timer = 0.0
        self.finished = False
        self.success = False

        # 3. Set Platform
        if fixed_platform_pos is not None:
            self.platform_pos = np.array(fixed_platform_pos)
        else:
            self._randomize_platform()

    def _randomize_platform(self):
        # Random r between 0.2 and 0.8 of radius, random theta
        r = np.random.uniform(0.2 * POOL_RADIUS_M, 0.8 * POOL_RADIUS_M)
        theta = np.random.uniform(0, 2 * np.pi)
        self.platform_pos = np.array([r * np.cos(theta), r * np.sin(theta)])

    def update(self, input_vector):
        """
        input_vector: Normalized vector (x, y) from keyboard input.
        """
        if self.finished:
            return

        self.timer += DT

        # 1. Calculate Target Velocity based on Input
        # If input, target is max speed. If no input, target is 0 (friction).
        target_velocity = input_vector * RAT_SPEED_M_S

        # 2. Apply Momentum (Foster Morris Dayan Rule)
        # v_t = (v_input + 3 * v_prev) / 4
        # We integrate this over DT to smooth it for high FPS
        
        # Note: The paper uses discrete steps. For 60FPS continuous gameplay, 
        # we interpolate towards the target velocity to simulate that "heavy" feeling.
        # The higher the momentum ratio, the slower it changes direction.
        alpha = 1.0 / (1.0 + MOMENTUM_WEIGHT_RATIO) 
        # Adjust alpha for Time Step to behave like the discrete model
        # Roughly: We want to match the paper's inertia feel.
        self.velocity = self.velocity * (1 - alpha) + target_velocity * alpha

        # 3. Move
        step = self.velocity * DT
        new_pos = self.pos + step

        # 4. Wall Collision (Reflection)
        dist_sq = np.dot(new_pos, new_pos)
        if dist_sq > POOL_RADIUS_M**2:
            # Hit the wall
            dist = np.sqrt(dist_sq)
            normal = new_pos / dist # Vector pointing out from center
            
            # Reflect velocity vector: v_new = v_old - 2(v_old . n)n
            # This creates the "bounce" effect described in the paper
            v_dot_n = np.dot(self.velocity, normal)
            self.velocity = self.velocity - 2 * v_dot_n * normal
            
            # Push back inside
            overlap = dist - POOL_RADIUS_M
            new_pos = new_pos - normal * (overlap * 1.1) 

        self.pos = new_pos

        # 5. Check Goal
        dist_to_plat = np.linalg.norm(self.pos - self.platform_pos)
        if dist_to_plat < PLATFORM_RADIUS_M:
            self.finished = True
            self.success = True
        
        # 6. Check Timeout
        if self.timer >= TIMEOUT_SECONDS:
            self.finished = True
            self.success = False

# ==============================================================================
# 3. GAME ENGINE (PYGAME)
# ==============================================================================

class RatGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("IAR Project - Watermaze Playable")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.large_font = pygame.font.SysFont("Arial", 32)
        
        self.physics = WaterMazePhysics()
        self.state = "MENU" # MENU, PLAY, INTERSTITIAL, GAMEOVER
        
        # Task State
        self.mode = None # "RMW", "DMP", "DEBUG"
        self.day = 1
        self.trial = 1
        self.current_platform_fixed = None # Used for RMW/DMP consistency

        # Trial Tracking
        self.trial_times = {}  # {day: [trial_times]}
        self.trial_successes = {}  # {day: [success_flags]}

        # Visuals
        self.center = np.array([SCREEN_SIZE // 2, SCREEN_SIZE // 2])

    def to_screen(self, pos_meters):
        """Convert meters (physics) to pixels (screen). Y is inverted in Pygame."""
        px = int(self.center[0] + pos_meters[0] * PIXELS_PER_METER)
        py = int(self.center[1] - pos_meters[1] * PIXELS_PER_METER)
        return (px, py)

    def run(self):
        running = True
        while running:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    self.handle_input(event.key)

            # Update & Draw
            self.screen.fill(COLOR_BG)

            if self.state == "MENU":
                self.draw_menu()
            elif self.state == "PLAY":
                self.update_physics()
                self.draw_game()
            elif self.state == "INTERSTITIAL":
                self.draw_interstitial()
            elif self.state == "GAMEOVER":
                self.draw_gameover()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    def handle_input(self, key):
        if self.state == "MENU":
            if key == pygame.K_1:
                self.start_task("RMW")
            elif key == pygame.K_2:
                self.start_task("DMP")
            elif key == pygame.K_3:
                self.start_task("DEBUG")
        
        elif self.state == "INTERSTITIAL":
            if key == pygame.K_SPACE:
                self.next_trial()

        elif self.state == "GAMEOVER":
            if key == pygame.K_SPACE:
                self.state = "MENU"

    def plot_trial_times(self):
        """Display and save a plot of trial times similar to Figure 4 format."""
        if not self.trial_times:
            return
        
        # Prepare data
        day_count = len(self.trial_times)
        trials_per_day = TRIALS_PER_DAY
        
        # Convert dict to arrays
        trial_times_array = []
        trial_successes_array = []
        
        for day in range(1, day_count + 1):
            if day in self.trial_times:
                trial_times_array.append(self.trial_times[day])
                trial_successes_array.append(self.trial_successes[day])
            else:
                trial_times_array.append([])
                trial_successes_array.append([])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # --- Plot 1: Trial Times ---
        space_between_days = 2
        x_positions = []
        day_positions = []
        all_times = []
        
        for day in range(day_count):
            day_x_positions = []
            for trial in range(len(trial_times_array[day])):
                pos = day * (trials_per_day + space_between_days) + trial + 1
                x_positions.append(pos)
                day_x_positions.append(pos)
                all_times.append(trial_times_array[day][trial])
            day_positions.append(day_x_positions)
        
        # Plot trial times
        for day, day_x_positions in enumerate(day_positions):
            if day_x_positions:
                ax1.plot(day_x_positions, trial_times_array[day], 
                        marker='o', color="black", linestyle='-', linewidth=2)
        
        # Day ticks for plot 1
        major_ticks = [(day * (trials_per_day + space_between_days)) + trials_per_day / 2 
                       for day in range(day_count)]
        ax1.set_xticks(major_ticks)
        ax1.set_xticklabels([str(day + 1) for day in range(day_count)])
        ax1.set_xlabel("Day", fontsize=12)
        ax1.set_ylabel("Trial Time (seconds)", fontsize=12)
        ax1.set_title(f"Trial Completion Times - {self.mode} Task", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # --- Plot 2: Success Rate ---
        success_rates = []
        for day in range(day_count):
            if trial_successes_array[day]:
                success_rate = sum(trial_successes_array[day]) / len(trial_successes_array[day]) * 100
                success_rates.append(success_rate)
            else:
                success_rates.append(0)
        
        days = np.arange(1, day_count + 1)
        colors = ['green' if sr == 100 else 'orange' if sr > 50 else 'red' for sr in success_rates]
        ax2.bar(days, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_xticks(days)
        ax2.set_xticklabels([str(d) for d in days])
        ax2.set_xlabel("Day", fontsize=12)
        ax2.set_ylabel("Success Rate (%)", fontsize=12)
        ax2.set_title(f"Daily Success Rate - {self.mode} Task", fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"trial_results_{self.mode}_{timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        
        # Display plot
        plt.show()

    def handle_input(self, key):
        if self.state == "MENU":
            if key == pygame.K_1:
                self.start_task("RMW")
            elif key == pygame.K_2:
                self.start_task("DMP")
            elif key == pygame.K_3:
                self.start_task("DEBUG")
        
        elif self.state == "INTERSTITIAL":
            if key == pygame.K_SPACE:
                self.next_trial()

        elif self.state == "GAMEOVER":
            if key == pygame.K_SPACE:
                self.state = "MENU"

    def start_task(self, mode):
        self.mode = mode
        self.day = 1
        self.trial = 1
        
        # Reset trial tracking
        self.trial_times = {}
        self.trial_successes = {}
        
        # Generate initial platform positions
        self.physics._randomize_platform() # Generate a random one
        
        if mode == "RMW":
            # RMW: Keep this position for days 1-7
            self.current_platform_fixed = self.physics.platform_pos.copy()
            # We also need a secondary position for Day 8 (DMP/Reversal)
            self.secondary_platform_pos = None 
        elif mode == "DMP":
            # DMP: Keep this position for Day 1 only
            self.current_platform_fixed = self.physics.platform_pos.copy()
        elif mode == "DEBUG":
            self.current_platform_fixed = self.physics.platform_pos.copy()

        self.physics.reset_trial(fixed_platform_pos=self.current_platform_fixed)
        self.state = "PLAY"

    def next_trial(self):
        # Record trial time and success
        if self.day not in self.trial_times:
            self.trial_times[self.day] = []
            self.trial_successes[self.day] = []
        
        self.trial_times[self.day].append(self.physics.timer)
        self.trial_successes[self.day].append(self.physics.success)
        
        # Advance counters
        self.trial += 1
        if self.trial > TRIALS_PER_DAY:
            self.trial = 1
            self.day += 1
        
        # Check Game Over
        max_days = 0
        if self.mode == "RMW": max_days = RMW_DAYS_PHASE_1 + RMW_DAYS_PHASE_2 + RMW_DAYS_PHASE_3
        elif self.mode == "DMP": max_days = DMP_TOTAL_DAYS
        elif self.mode == "DEBUG": max_days = 999

        if self.day > max_days:
            self.state = "GAMEOVER"
            # Display and save plot when task is complete
            self.plot_trial_times()
            return

        # Handle Platform Logic based on Day
        if self.mode == "DMP":
            # In DMP, platform moves every new day
            if self.trial == 1:
                print(f"DMP Day {self.day}: Changing platform location.")
                self.physics._randomize_platform()
                self.current_platform_fixed = self.physics.platform_pos.copy()
        
        elif self.mode == "RMW":
            # Phase 1: Days 1-7 (Goal A)
            if self.day <= RMW_DAYS_PHASE_1:
                pass # Use existing Goal A
            
            # Phase 2: Day 8 (Goal B - Reversal)
            elif self.day == RMW_DAYS_PHASE_1 + 1:
                if self.trial == 1:
                    print(f"RMW Day {RMW_DAYS_PHASE_1 + 1}: Changing platform location (Reversal).")
                    self.physics._randomize_platform()
                    self.secondary_platform_pos = self.physics.platform_pos.copy()
                self.current_platform_fixed = self.secondary_platform_pos
            
            # Phase 3: Day 9 (Stay on same goal (B))
            elif self.day == RMW_DAYS_PHASE_1 + 1 + 1:
                pass

        elif self.mode == "DEBUG":
            # Always random every trial
            self.physics._randomize_platform()
            self.current_platform_fixed = self.physics.platform_pos.copy()

        self.physics.reset_trial(fixed_platform_pos=self.current_platform_fixed)
        self.state = "PLAY"

    def update_physics(self):
        # 1. Get Keyboard Input
        keys = pygame.key.get_pressed()
        input_vec = np.array([0.0, 0.0])
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:    input_vec[1] += 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:  input_vec[1] -= 1
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:  input_vec[0] -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: input_vec[0] += 1

        # Normalize input
        norm = np.linalg.norm(input_vec)
        if norm > 0:
            input_vec /= norm

        # 2. Update Physics
        self.physics.update(input_vec)

        # 3. Check End State
        if self.physics.finished:
            self.state = "INTERSTITIAL"

    # ==========================================================================
    # DRAWING FUNCTIONS
    # ==========================================================================
    
    def draw_menu(self):
        title = self.large_font.render("Morris Water Maze Simulator", True, COLOR_TEXT)
        opt1 = self.font.render("1. Reference Memory Task (RMW)", True, COLOR_TEXT)
        opt2 = self.font.render("2. Delayed Matching to Place (DMP)", True, COLOR_TEXT)
        opt3 = self.font.render("3. Debug Mode (Visible Platform)", True, COLOR_TEXT)
        
        info = self.font.render("Controls: Arrow Keys to swim", True, (150, 150, 150))

        self.screen.blit(title, (SCREEN_SIZE//2 - title.get_width()//2, 100))
        self.screen.blit(opt1, (SCREEN_SIZE//2 - opt1.get_width()//2, 250))
        self.screen.blit(opt2, (SCREEN_SIZE//2 - opt2.get_width()//2, 300))
        self.screen.blit(opt3, (SCREEN_SIZE//2 - opt3.get_width()//2, 350))
        self.screen.blit(info, (SCREEN_SIZE//2 - info.get_width()//2, 500))

    def draw_game(self):
        # 1. Draw Pool
        pygame.draw.circle(self.screen, COLOR_POOL, self.to_screen([0,0]), int(POOL_RADIUS_M * PIXELS_PER_METER))
        pygame.draw.circle(self.screen, (200, 200, 200), self.to_screen([0,0]), int(POOL_RADIUS_M * PIXELS_PER_METER), 2) # Rim

        # 2. Draw Platform (Logic: Invisible unless Debug or Finished)
        should_draw_plat = False
        if self.mode == "DEBUG": should_draw_plat = True
        if self.physics.finished and self.physics.success: should_draw_plat = True
        
        if should_draw_plat:
            p_pos = self.to_screen(self.physics.platform_pos)
            p_rad = int(PLATFORM_RADIUS_M * PIXELS_PER_METER)
            
            # Create a transparent surface for the platform
            s = pygame.Surface((p_rad*2, p_rad*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (0, 255, 0, 128), (p_rad, p_rad), p_rad)
            self.screen.blit(s, (p_pos[0]-p_rad, p_pos[1]-p_rad))

        # 3. Draw Rat
        r_pos = self.to_screen(self.physics.pos)
        pygame.draw.circle(self.screen, COLOR_RAT, r_pos, 8)
        pygame.draw.circle(self.screen, (255, 255, 255), r_pos, 8, 2) # Outline

        # 4. Draw HUD
        hud_height = 80
        pygame.draw.rect(self.screen, COLOR_HUD_BG, (0, SCREEN_SIZE - hud_height, SCREEN_SIZE, hud_height))
        day_text = self.font.render(f"Day: {self.day}", True, COLOR_TEXT)
        trial_text = self.font.render(f"Trial: {self.trial}", True, COLOR_TEXT)
        time_text = self.font.render(f"Time: {self.physics.timer:.1f}s", True, COLOR_TEXT)
        self.screen.blit(day_text, (20, SCREEN_SIZE - hud_height + 10))
        self.screen.blit(trial_text, (20, SCREEN_SIZE - hud_height + 40))
        self.screen.blit(time_text, (200, SCREEN_SIZE - hud_height + 10))

    def draw_interstitial(self):
        msg = "Trial Complete! " 
        if self.physics.success:
            msg += "You found the platform!"
        else:
            msg += "Time's up!"
        msg += " Press SPACE to continue."
        
        text_surf = self.font.render(msg, True, COLOR_TEXT)
        self.screen.blit(text_surf, (SCREEN_SIZE//2 - text_surf.get_width()//2, SCREEN_SIZE//2 - text_surf.get_height()//2))

    def draw_gameover(self):
        msg = "Task Complete! Press SPACE to return to Menu."
        text_surf = self.font.render(msg, True, COLOR_TEXT)
        self.screen.blit(text_surf, (SCREEN_SIZE//2 - text_surf.get_width()//2, SCREEN_SIZE//2 - text_surf.get_height()//2))

if __name__ == "__main__":
    game = RatGame()
    game.run()
