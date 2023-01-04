import numpy as np
from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import pygame
import pickle
from main import *

class HeuristicController(FlightController):


    def __init__(self):
        """Creates a heuristic flight controller with some specified parameters

        """
        pass
        

    def get_max_simulation_steps(self):
            return 1000 # You can alter the amount of steps you want your program to run for here


    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        """Takes a given drone object, containing information about its current state
        and calculates a pair of thrust values for the left and right propellers.

        Args:
            drone (Drone): The drone object containing the information about the drones state.

        Returns:
            Tuple[float, float]: A pair of floating point values which respectively represent the thrust of the left and right propellers, must be between 0 and 1 inclusive.
        """

        target_point = drone.get_next_target()
        
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y
        
        self.dist_from_last_step = np.sqrt(dx ** 2 + dy ** 2 )

        thrust_adj = np.clip(dy * self.ky, -self.abs_thrust_delta, self.abs_thrust_delta)
        target_pitch = np.clip(dx * self.kx, -self.abs_pitch_delta, self.abs_pitch_delta)
        delta_pitch = target_pitch-drone.pitch

        thrust_left = np.clip(0.5 + thrust_adj + delta_pitch, 0.0, 1.0)
        thrust_right = np.clip(0.5 + thrust_adj - delta_pitch, 0.0, 1.0)

        # The default controller sets each propeller to a value of 0.5 0.5 to stay stationary.
        return (thrust_left, thrust_right)

    def train(self):
        """A self contained method designed to train parameters created in the initialiser.
        """
        epochs = 10
        params = [1.0, 0.5, 0.1, 0.3]
        alpha = 1e-4
        
        grads = [0, 0, 0, 0]

        for n in range(epochs):
            
            ## Forward pass
            print(f"Forward pass for epoch:{n+1}")        
            R1 = self.getReward(params)
            print(f"Reward: {R1}")
            ## Log the data before backprop
            data = {'epoch':n+1, 'params':params, 'grads':grads, 'reward': R1}            
            self.logWeights(n+1, data)            
            ## Estimate grads / for backprop
            print(f"Back prop for epoch:{n+1}")
                        
            for i in range(len(params)):
                print(f" -- Finding Grad for parameter {i+1}")
                delta = 0.0001
                p1 = params[i]
                p2 = p1 + delta
                params_changed = params.copy()
                params_changed[i] = p2
                
                #R1 = self.getReward(params)
                R2 = self.getReward(params_changed)
                #Calclate the gradient
                dR = (R2-R1)/(p2-p1)
                grads[i] = dR
                print(f"   - Reward is: {R2} Grad is: {dR}")
            
            ## Update parameters
            for i in range(len(params)):
                params[i] = params[i] + alpha * grads[i]           
            print(f"Updated params :{params}")
            print("\n\n")

            
        print("Finding Final reward")
        R1 = self.getReward(params)
        print(f"Final Reward: {R1}")
        print(f"Final params: {params}")
        ## Log the data of final run
        data = {'epoch':n+1, 'params':params, 'grads':grads, 'reward': R1}
        self.logWeights(n+1, data)
        
    def getReward(self, params):
        
        self.ky = params[0]
        self.kx = params[1]
        self.abs_pitch_delta = params[2]
        self.abs_thrust_delta = params[3]
        reward = 0
        last_distance = float("inf")
        
        # Initialise pygame
        pygame.init()
        clock = pygame.time.Clock()

        # Load the relevant graphics into pygame
        drone_img = pygame.image.load('graphics/drone_small.png')
        background_img = pygame.image.load('graphics/background.png')
        target_img = pygame.image.load('graphics/target.png')
    

        # Create the screen
        SCREEN_WIDTH = 720
        SCREEN_HEIGHT = 480
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))    
        delta_time = self.get_time_interval()           
           
            
        # create a new drone simulation
        drone = self.init_drone()
        # 3) run simulation
        for t in range(self.get_max_simulation_steps()):
            drone.set_thrust(self.get_thrusts(drone))
            drone.step_simulation(self.get_time_interval())
            # Refresh the background
            screen.blit(background_img, (0,0))
            # Draw the current drone on the screen
            draw_drone(screen, drone, drone_img)
            # Draw the next target on the screen
            draw_target(drone.get_next_target(), screen, target_img)
        
            # Actually displays the final frame on the screen
            pygame.display.flip()
        
            # Makes sure that the simulation runs at a target 60FPS
            clock.tick(60)
                
            # Calculate reward
            if drone.has_reached_target_last_update:                    
                reward += 2
                last_distance = float("inf")
                                      
            elif self.findDistance(drone) < last_distance:
                last_distance = self.findDistance(drone)
                reward += 1
                
            else:
                reward -= 1          
            
            
        return reward
        
    def logWeights(self, epoch, data):
        file_name = 'weights/current/params_epoch{}.pkl'.format(epoch)        
        with open(file_name,'wb') as file:
            pickle.dump(data, file)
        
    def findDistance(self, drone: Drone):
        target_point = drone.get_next_target()
        
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y
        
        return np.sqrt(dx ** 2 + dy ** 2 )

    def load(self):
        """Load the parameters of this flight controller from disk.
        """
        try:
            parameter_array = np.load('heuristic_controller_parameters.npy')
            self.ky = parameter_array[0]
            self.kx = parameter_array[1]
            self.abs_pitch_delta = parameter_array[2]
            self.abs_thrust_delta = parameter_array[3]
        except:
            print("Could not load parameters, sticking with default parameters.")

    def save(self):
        """Save the parameters of this flight controller to disk.
        """
        parameter_array = np.array([self.ky, self.kx, self.abs_pitch_delta, self.abs_thrust_delta])
        np.save('heuristic_controller_parameters.npy', parameter_array)
        
        
if __name__ == "__main__":
    controller = HeuristicController()
    controller.train()