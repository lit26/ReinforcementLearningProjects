# Author: Shiva Verma
import turtle as t


class Paddle():
    def __init__(self):
        self.hit, self.miss = 0, 0

        # Setup Background
        self.win = t.Screen()
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)

        # Paddle
        self.paddle = t.Turtle()
        self.paddle.speed(0)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, -275)

        # Ball
        self.ball = t.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('red')
        self.ball.penup()
        self.ball.goto(0, 100)
        self.ball.dx = 4
        self.ball.dy = -3

        # Score
        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        # -------------------- Keyboard control ----------------------
        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # Paddle movement
    def paddle_right(self):
        x = self.paddle.xcor()
        if x < 225:
            self.paddle.setx(x+20)

    def paddle_left(self):
        x = self.paddle.xcor()
        if x > -225:
            self.paddle.setx(x-20)

    def movement(self, action):
        x = self.paddle.xcor()
        if action =='left' and x > -225:
            self.paddle.setx(x - 20)
        elif action == 'right' and x < 225:
            self.paddle.setx(x + 20)

    def reset(self):
        self.paddle.goto(0, -275)
        self.ball.goto(0, 100)

    def restart(self):
        self.reset()
        self.hit, self.miss = 0, 0

    def run_frame(self):
        done = False
        hit = False
        self.win.update()

        # Ball moving
        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision
        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground contact
        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            done = True

        # Ball Paddle collision
        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            hit = True
        return hit, done

if __name__=="__main__":
    game = Paddle()
    while True:
        game.run_frame()
