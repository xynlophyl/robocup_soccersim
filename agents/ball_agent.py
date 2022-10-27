import random
from soccerpy.agent import Agent as BaseAgent
from soccerpy.world_model import WorldModel

class Agent(BaseAgent):
    """
    A simple Agent class that follows the ball and kicks it forward
    """

    def think(self):
        """
        Performs a single step of thinking for our agent. Gets called on every
        iteration of our think loop. Uses modification from kengz that lets agents think whenever ball is in play or our team is kicking off
        """

        # DEBUG:  tells us if a thread dies
        if not self.__think_thread.is_alive() or not self.__msg_thread.is_alive():
            raise Exception("A thread died.")

        # take places on the field by uniform number
        if not self.in_kick_off_formation:
            print("the side is", self.wm.side)

            # used to flip x coords for other side
            side_mod = 1
            if self.wm.side == WorldModel.SIDE_R:
                side_mod = -1

            if self.wm.uniform_number == 1:
                self.wm.teleport_to_point((-5 * side_mod, 30))
            elif self.wm.uniform_number == 2:
                self.wm.teleport_to_point((-40 * side_mod, 15))
            elif self.wm.uniform_number == 3:
                self.wm.teleport_to_point((-40 * side_mod, 00))
            elif self.wm.uniform_number == 4:
                self.wm.teleport_to_point((-40 * side_mod, -15))
            elif self.wm.uniform_number == 5:
                self.wm.teleport_to_point((-5 * side_mod, -30))
            elif self.wm.uniform_number == 6:
                self.wm.teleport_to_point((-20 * side_mod, 20))
            elif self.wm.uniform_number == 7:
                self.wm.teleport_to_point((-20 * side_mod, 0))
            elif self.wm.uniform_number == 8:
                self.wm.teleport_to_point((-20 * side_mod, -20))
            elif self.wm.uniform_number == 9:
                self.wm.teleport_to_point((-10 * side_mod, 0))
            elif self.wm.uniform_number == 10:
                self.wm.teleport_to_point((-10 * side_mod, 20))
            elif self.wm.uniform_number == 11:
                self.wm.teleport_to_point((-10 * side_mod, -20))

            self.in_kick_off_formation = True

            return

        # determine the enemy goal position
        # self.enemy_goal_pos = None
        # self.own_goal_pos = None
        if self.wm.side == WorldModel.SIDE_R:
            self.enemy_goal_pos = (-55, 0)
            self.own_goal_pos = (55, 0)
        else:
            self.enemy_goal_pos = (55, 0)
            self.own_goal_pos = (-55, 0)

        if not self.wm.is_before_kick_off() or self.wm.is_kick_off_us() or self.wm.is_playon():
            # The main decision loop
            return self.decisionLoop()

    def find_ball(self):
        # find the ball
        if self.wm.ball is None or self.wm.ball.direction is None:
            self.wm.ah.turn(30)
            if not -7 <= self.wm.ball.direction <= 7:
                self.wm.ah.turn(self.wm.ball.direction / 2)

            return

    def move_to_ball(self):
        print('move_to_ball')
        self.wm.ah.dash(60)
        return

    def dribble(self):
        print("dribbling")
        pos = (random.randint(1,50), random.randint(1,50))
        # self.wm.kick_to(self.enemy_goal_pos, 1.0)
        # self.wm.turn_body_to_point(self.enemy_goal_pos)

        self.wm.kick_to(pos, extra_power=4.0)
        self.wm.turn_body_to_point(pos)

        self.wm.align_neck_with_body()
        self.wm.ah.dash(50)
        return
    
    def shall_dribble(self):
        # find the ball
        # self.find_ball()
        # if self.wm.ball is None or self.wm.ball.direction is None:
            # self.wm.ah.turn(30)
        return self.wm.is_ball_kickable()
    
    def shall_move_to_ball(self):
        return not(self.wm.ball is None or self.wm.ball.direction is None)

    def defaultaction(self):
        # print "def"
        # kick off!
        if self.wm.is_before_kick_off():
            # player 9 takes the kick off
            if self.wm.uniform_number == 9:
                if self.wm.is_ball_kickable():
                    # kick with 100% extra effort at enemy goal
                    self.wm.kick_to(self.enemy_goal_pos, 1.0)
                else:
                    # move towards ball
                    if self.wm.ball is not None:
                        if (self.wm.ball.direction is not None and
                                -7 <= self.wm.ball.direction <= 7):
                            self.wm.ah.dash(50)
                        else:
                            self.wm.turn_body_to_point((0, 0))

                # turn to ball if we can see it, else face the enemy goal
                if self.wm.ball is not None:
                    self.wm.turn_neck_to_object(self.wm.ball)

                return

    def decisionLoop(self):
        try:
            self.find_ball() # tries to find the ball
            if self.shall_dribble(): # if agent has the ball, dribble
                return self.dribble()
            elif self.shall_move_to_ball(): # if agent knows where the ball is, move to it
                return self.move_to_ball()
            else: # do default action
                return self.defaultaction()
        except:
            self.defaultaction()