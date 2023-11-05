import pygame
#引入pygame中所有常量，比如 QUIT
from pygame.locals import *
pygame.init()
screen = pygame.display.set_mode((1000,600))
pygame.display.set_caption('plants vs zombies')

image_surface = pygame.image.load("C:/Users/23958/Desktop/pool.jpg")

#image_surface.fill((0,0,255),rect=(100,100,100,50),special_flags=0)

image_surface.scroll(-50,0)

image_wallnut = pygame.image.load("C:/Users/23958/Desktop/wallnut.gif")
image_wallnut2 = pygame.transform.rotate(image_wallnut,45)
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            exit()
    # 将图像放置在主屏幕上
    screen.blit(image_surface,(0,0))
    screen.blit(image_wallnut,(800,500))
    pygame.display.update()