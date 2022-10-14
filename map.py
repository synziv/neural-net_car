from pyglet import shapes
#obstacle

obstacles = []
class obstacle:
    def __init__(self, x, y, width, height, batch):
        self.sprite = shapes.Rectangle(x, y, width, height, color=(255, 22, 20), batch=batch)
        self.obstacle_lines = [
            [(self.sprite.x, self.sprite.y), (self.sprite.x + self.sprite.width, self.sprite.y)],
            [(self.sprite.x + self.sprite.width, self.sprite.y), (self.sprite.x + self.sprite.width, self.sprite.y + self.sprite.height)],
            [(self.sprite.x + self.sprite.width, self.sprite.y + self.sprite.height), (self.sprite.x, self.sprite.y + self.sprite.height)],
            [(self.sprite.x, self.sprite.y + self.sprite.height), (self.sprite.x, self.sprite.y)]
        ]

def initObstacle(batch):
    global obstacles
    obstacles = [
        obstacle(0, 150, 350, 20, batch),
        obstacle(300, 350, 350, 20, batch),
    ]