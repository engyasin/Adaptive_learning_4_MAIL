from gym_multigrid.multigrid import *
import numpy as np
import time

class SoccerGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to fetch the balls and drop them in their respective goals
    """

    def __init__(
        self,
        size=10,
        view_size=5,
        width=None,
        height=None,
        goal_pst = [],
        goal_index = [],
        num_balls=[],
        agents_index = [],
        balls_index=[],
        zero_sum = False,
        seed=1000,

    ):
        self.num_balls = num_balls
        self.goal_pst = goal_pst
        self.goal_index = goal_index
        self.balls_index = balls_index
        self.zero_sum = zero_sum

        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            seed=seed
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for i in range(len(self.goal_pst)):
            self.place_obj(ObjectGoal(self.world,self.goal_index[i], 'ball'), top=self.goal_pst[i], size=[1,1])

        for number, index in zip(self.num_balls, self.balls_index):
            for i in range(number):
                self.ball = Ball(self.world,index)
                self.place_obj(self.ball)

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reward(self, i, rewards,reward=1):
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    #if fwd_cell.type == 'ball':
                        #self._reward(self.agents[i].index, rewards, 0.01)
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            elif fwd_cell.type=='agent':
                if fwd_cell.carrying:
                    if self.agents[i].carrying is None:
                        self.agents[i].carrying = fwd_cell.carrying
                        #if fwd_cell.carrying.type == 'ball':
                            #self._reward(self.agents[i].index, rewards, 0.01)
                        fwd_cell.carrying = None

    def _handle_attack(self,i,rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            # it only can carry ball 
            agent_index = self.agents[i].index
            agent_pos = self.agents[i].pos
            fwd_pos = np.array(fwd_pos)
            # NOTE hard coded because we have only two teams
            opponent_goal_pos = np.array(self.goal_pst[agent_index%2])
            curr_dist = np.linalg.norm(agent_pos - opponent_goal_pos)
            new_dist = np.linalg.norm(fwd_pos - opponent_goal_pos)
            if new_dist < curr_dist:
                self._reward(self.agents[i].index, rewards, 0.01)
            elif new_dist > curr_dist:
                self._reward(self.agents[i].index, rewards, -0.01)

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            if fwd_cell:
                if fwd_cell.type == 'objgoal' and fwd_cell.target_type == self.agents[i].carrying.type:
                    if self.agents[i].carrying.index in [0, fwd_cell.index]:
                        #self._reward(fwd_cell.index, rewards, fwd_cell.reward)
                        self._reward(fwd_cell.index, rewards, -1)
                        # TODO : give another ball to the opponent team or randomly
                        for index in self.balls_index:
                            self.reinit_place(self.ball)
                            #self.place_obj(Ball(self.world,index))
                        self.agents[i].carrying = None
                elif fwd_cell.type=='agent':
                    if fwd_cell.carrying is None:
                        fwd_cell.carrying = self.agents[i].carrying
                        self.agents[i].carrying = None
            else:
                self.grid.set(*fwd_pos, self.agents[i].carrying)
                #if droped elsewhere
                #self._reward(self.agents[i].index, rewards, -0.005)
                self.agents[i].carrying.cur_pos = fwd_pos
                self.agents[i].carrying = None


    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info

    def reinit_place(self,obj):
        pos = obj.init_pos.copy()
        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

class SoccerGame4HEnv10x15N2(SoccerGameEnv):
    def __init__(self):
        super().__init__(size=None,
        height=10,
        width=15,
        goal_pst = [[1,5], [13,5]],
        goal_index = [1,2],
        num_balls=[1],
        agents_index = [1,1,2,2],
        balls_index=[0],
        zero_sum=True,
        seed=int(time.time()),)
