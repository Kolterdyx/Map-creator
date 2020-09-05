#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Ciro García <kolterdyx>
# @Date:   17-Aug-2020
# @Email:  kolterdev@gmail.com
# @Last modified by:   kolterdyx
# @Last modified time: 20-Aug-2020
# @License: This file is subject to the terms and conditions defined in file 'LICENSE', which is part of this source code package.

import os
import signal
import sys
import numpy as np
from multiprocessing import Process, Manager, Value, Array
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import pygame as pg

vec = pg.math.Vector2

manager = Manager()

SETTINGS = manager.dict()

start_settings = Value('i', 0)
in_settings = Value('i', 0)
quit_settings = Value('i', 0)
reset_settings = Value('i', 0)
generation_finished = Value('i', 0)
generation_started = Value('i', 0)
chunks_generated = Value('i', 0)
apply_settings = Value('i', 0)


SETTINGS["oct"] = 1
SETTINGS["scl"] = 250
SETTINGS["sea"] = 60
SETTINGS["lac"] = 2
SETTINGS["seed"] = 0
SETTINGS["wdh"] = 600
SETTINGS["hgt"] = 600
SETTINGS["offset"] = [0, 0]
SETTINGS["mwd"] = 600
SETTINGS["mhg"] = 600
SETTINGS["zoom"] = 0
SETTINGS["zoom_slider"] = 0
SETTINGS["noise"] = 2

SHARED_MAP = manager.dict()


def remap(OldValue, OldMin, OldMax, NewMin, NewMax):
    OldRange = (float(OldMax) - float(OldMin))
    NewRange = (float(NewMax) - float(NewMin))
    NewValue = (((float(OldValue) - float(OldMin)) * NewRange) / OldRange) + float(NewMin)
    return NewValue


def create_noise(iterations, x, y, scale, lacunarity, low, high, ngen):
    maxAmp = 0
    amp = 1
    persistence = 0.5
    freq = scale
    noise = 0

    for i in range(iterations):
        noise = noise + ngen.noise2d(x/freq, y/freq) * amp
        maxAmp = maxAmp + amp
        amp = amp * persistence
        freq = freq / lacunarity

    noise = noise / maxAmp

    noise = noise * (high - low) / 2 + (high + low) / 2

    return int(high-noise)


def draw_colored(octv, scale, sea, width, height, lac, pos, seed, noise, offset):

    from noise import pnoise2, snoise2
    from opensimplex import OpenSimplex
    import numpy as np

    generation_started.value = 1
    xoffset = offset[0] + pos[0]
    yoffset = offset[1] + pos[1]
    world = np.full((height, width, 3), 255, dtype=np.uint8)
    normal_world = world.copy()
    # Color all pixels
    for x in range(width):
        # print(x)
        for y in range(height):
            if noise == 2:
                n = remap(snoise2((x+xoffset)/scale, (y+yoffset)/scale,
                                  octaves=octv, base=seed, lacunarity=lac), -1, 1, 0, 100)
            elif noise == 1:
                n = remap(pnoise2((x+xoffset)/scale, (y+yoffset)/scale,
                                  octaves=octv, base=seed, lacunarity=lac), -1, 1, 0, 100)
            elif noise == 0:
                ngen = OpenSimplex(seed)
                n = create_noise(octv, x+xoffset, y+yoffset, scale, lac, 0, 100, ngen)

            color_dist = int((100-sea)/(octv+2))

            if n >= sea:
                r = range(sea, 100, color_dist)
                hmap = []
                cmap = []
                for i in r:
                    hmap.append(i)
                    cmap.append(remap(i, sea, 100, 150, 255))

                d = 255-max(cmap)

                for i, h in enumerate(hmap):
                    if n >= h:
                        normal_world[y][x][0] = cmap[i]/6
                        normal_world[y][x][1] = cmap[i] + d
                        normal_world[y][x][2] = cmap[i]/2

                        world[y][x][0] = cmap[i]/2
                        world[y][x][1] = cmap[i]
                        world[y][x][2] = cmap[i]/3

            else:
                r = range(0, sea, color_dist*2+1)
                hmap = []
                cmap = []
                for i in r:
                    hmap.append(i)
                    cmap.append(remap(i, 0, sea, 50, 150))

                for i, h in enumerate(hmap):
                    if n >= h:
                        normal_world[y][x][0] = 0
                        normal_world[y][x][1] = 0
                        normal_world[y][x][2] = cmap[i]

                        world[y][x][0] = cmap[i]
                        world[y][x][1] = 0
                        world[y][x][2] = 0

    SHARED_MAP[tuple(pos)] = normal_world
    generation_started.value = 0
    generation_finished.value = 1
    chunks_generated.value += 1


def sw():

    import pygame as pg
    from pgui import Slider, Entry, Button, CheckBox
    import json
    import os

    pg.init()
    pg.display.set_caption("Settings")

    class CustomCheckBox:
        def __init__(self, parent, *, x=0, y=0, size=20):
            self.parent = parent
            self.screen = parent.screen
            self.box = CheckBox(self, x=x, y=y, size=size)
            # ------- ATTRIBUTES -------
            # Set the default background color
            self.bg_color = (255, 255, 255)
            # Set the default border width
            self.border_width = 3
            # Set the default border color
            self.border_color = (0, 0, 0)
            # Set the default check mark color
            self.check_color = (0, 200, 0)
            # Set the 'cross' style check mark width
            self.cross_width = 5
            # self.checked will be True only when the box is checked
            self.checked = False
            # Set the default box style and check style
            self.check_style = "fill"
            # Set the default side for the text
            self.label_side = "top"
            # Set the default alignment of the text
            self.label_align = "left"
            self.label_padding = 3
            # --------------------------

        def update(self):
            self.group = self.parent.noise_options
            if self.box.clicked:
                self.checked = True
                for c in self.group:
                    if c != self:
                        c.checked = False
            else:
                pass
            self.sync_attributes()
            self.box.update()

        def sync_attributes(self):
            # ------- ATTRIBUTES -------
            self.box.bg_color = self.bg_color
            self.box.border_width = self.border_width
            self.box.border_color = self.border_color
            self.box.check_color = self.check_color
            self.box.cross_width = self.cross_width
            self.box.checked = self.checked
            self.box.check_style = self.check_style
            self.box.label_side = self.label_side
            self.box.label_align = self.label_align
            self.box.label_padding = self.label_padding
            # --------------------------

        def move(self, x, y):
            self.box.move(x, y)

        def set_font(self, font):
            self.box.set_font(font)

        def set_font_size(self, size):
            self.box.set_font_size(size)

        def set_font_color(self, color):
            self.box.set_font_color(color)

        def set_size(self, size):
            self.box.set_size(size)

        def set_style(self, style):
            self.box.set_style(style)

        def set_label(self, text):
            self.box.set_label(text)

    class Screen:
        def create(self):

            self.screen = pg.display.set_mode((800, 600))

            self.load_button = Button(self, text="Load config", func=self.load)
            self.load_button.move(20, 555)
            self.load_button.height = 25
            self.load_button.set_font_size(15)

            self.save_button = Button(self, text="Save config", func=self.save)
            self.save_button.move(145, 555)
            self.save_button.height = 25
            self.save_button.set_font_size(15)

            self.reset_button = Button(self, text="Reset config", func=self.reset)
            self.reset_button.move(270, 555)
            self.reset_button.height = 25
            self.reset_button.set_font_size(15)

            self.opensimplex_option = CustomCheckBox(self, x=20, y=20)
            self.opensimplex_option.check_color = (0, 0, 0)
            self.opensimplex_option.set_font_size(15)
            self.opensimplex_option.set_label("OpenSimplex noise")
            self.opensimplex_option.label_side = "right"
            self.opensimplex_option.label_padding = 5

            self.perlin_noise_option = CustomCheckBox(self, x=20, y=50)
            self.perlin_noise_option.check_color = (0, 0, 0)
            self.perlin_noise_option.set_font_size(15)
            self.perlin_noise_option.set_label("Perlin noise")
            self.perlin_noise_option.label_side = "right"
            self.perlin_noise_option.label_padding = 5

            self.simplex_option = CustomCheckBox(self, x=20, y=80)
            self.simplex_option.check_color = (0, 0, 0)
            self.simplex_option.set_font_size(15)
            self.simplex_option.set_label("Simplex noise")
            self.simplex_option.label_side = "right"
            self.simplex_option.label_padding = 5

            self.settings_file = Entry(self, x=20, y=500, border=1, size=15, width=300)
            self.settings_file.width = 300
            self.settings_file.text = "settings.json"
            self.settings_file.set_label("Settings file:")

            self.noise_options = [
                self.opensimplex_option,
                self.perlin_noise_option,
                self.simplex_option
            ]

            self.noise_options[SETTINGS["noise"]].checked = True

            self.widgets = [
                self.load_button,
                self.save_button,
                self.reset_button,
                self.opensimplex_option,
                self.perlin_noise_option,
                self.simplex_option,
                self.settings_file
            ]

            for w in self.widgets:
                w.border_width = 1

        def load(self):
            filename = self.settings_file.text
            if os.path.isfile(filename):
                with open(filename, 'r') as f:
                    settings = json.load(f)

                SETTINGS.update(settings)

                self.apply_settings()
                self.settings_file.set_label("Settings file:")
            else:
                self.settings_file.set_label("Settings file:    (file not found)")

        def save(self):
            settings = SETTINGS.copy()
            filename = "settings.json"
            with open(filename, 'w') as f:
                json.dump(settings, f)

        def apply_settings(self):
            apply_settings.value = 1
            for i in self.noise_options:
                i.checked = False
            self.noise_options[SETTINGS["noise"]].checked = True

        def reset(self):
            reset_settings.value = 1
            SETTINGS["noise"] = 2
            self.noise_options[2].checked = True
            self.noise_options[0].checked = False
            self.noise_options[1].checked = False

        def run(self):
            if start_settings.value:
                self.create()
                start_settings.value = 0
                in_settings.value = 1
            if in_settings.value:
                self.update()
                self.events()
            if quit_settings.value:
                raise SystemExit

        def update(self):
            self.screen.fill((200, 200, 200))
            for w in self.widgets:
                w.update()

            noises = [
                self.opensimplex_option.checked,
                self.perlin_noise_option.checked,
                self.simplex_option.checked
            ]

            for i, n in enumerate(noises):
                if noises[i]:
                    SETTINGS["noise"] = i

            pg.display.flip()

        def events(self):
            if in_settings.value:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        in_settings.value = 0
                        pg.display.quit()
                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_ESCAPE:
                            in_settings.value = 0
                            pg.display.quit()

    sc = Screen()
    while True:
        sc.run()


def editor():
    import pygame as pg
    from pgui import Slider, Entry, Button, CheckBox
    import os
    import cv2

    pg.init()

    screen = pg.display.set_mode((1200, 704))
    pg.display.set_caption("Map creator")

    class Editor:
        def __init__(self):
            self.screen = screen
            self.value = 5

            self.oct_slider = Slider(self, max=7)
            self.oct_slider.move(20, 40)
            self.oct_slider.set_length(300)
            self.oct_slider.set_font_size(15)
            self.oct_slider.set_width(15)
            self.oct_slider.set_label("Octaves: 1")
            self.oct_slider.set_mark(0)

            self.sea_level_slider = Slider(self)
            self.sea_level_slider.move(20, 80)
            self.sea_level_slider.set_length(300)
            self.sea_level_slider.set_font_size(15)
            self.sea_level_slider.set_width(15)
            self.sea_level_slider.set_label("Sea level: 60")
            self.sea_level_slider.set_mark(60)

            self.scale_slider = Slider(self, max=599)
            self.scale_slider.move(575, 660)
            self.scale_slider.set_length(585)
            self.scale_slider.set_font_size(15)
            self.scale_slider.set_width(15)
            self.scale_slider.set_label("Scale: 100")
            self.scale_slider.set_mark(99)

            self.zoom_slider = Slider(self, max=200, orientation="vertical")
            self.zoom_slider.move(525, 25)
            self.zoom_slider.set_length(585)
            self.zoom_slider.label_align = "center"
            self.zoom_slider.set_font_size(15)
            self.zoom_slider.set_label("Zoom: 1")
            self.zoom_slider.set_width(15)
            self.zoom_slider.set_mark(175)
            self.zoom = 1

            self.map_surface = pg.Surface((600, 600))
            self.map_surface.fill(0)
            self.map_rect = self.map_surface.get_rect()
            self.map_rect.topleft = (575, 25)
            self.map = pg.Surface((SETTINGS["wdh"], SETTINGS["hgt"]))
            self.map.fill((100, 100, 100))

            self.width_slider = Slider(self, max=120)
            self.width_slider.move(20, 120)
            self.width_slider.set_length(300)
            self.width_slider.set_font_size(15)
            self.width_slider.set_width(15)
            self.width_slider.set_label("Width: 600")
            self.width_slider.set_mark(50)

            self.height_slider = Slider(self, max=120)
            self.height_slider.move(20, 160)
            self.height_slider.set_length(300)
            self.height_slider.set_font_size(15)
            self.height_slider.set_width(15)
            self.height_slider.set_label("Height: 600")
            self.height_slider.set_mark(50)

            self.pos_entry = Entry(self, x=20, y=200, border=1, size=15)
            self.pos_entry.text = '0,0'
            self.pos_entry.set_label("X,Y Position:")

            self.lac_slider = Slider(self, max=40)
            self.lac_slider.move(20, 240)
            self.lac_slider.set_length(300)
            self.lac_slider.set_font_size(15)
            self.lac_slider.set_width(15)
            self.lac_slider.set_label("Lacunarity: 2")
            self.lac_slider.set_mark(10)

            self.seed_entry = Entry(self, x=20, y=360, border=1, size=15)
            self.seed_entry.text = '0'
            self.seed_entry.set_label("Seed:")

            self.draw_button = Button(self, x=20, y=625, text="Generate", func=self.draw)
            self.draw_button.width = 315

            self.clear_button = Button(self, x=345, y=625, text="Clear", func=self.clear)

            self.settings_button = Button(self, x=20, y=570, text='Settings', func=self.settings)
            self.settings_button.height = 25
            self.settings_button.set_font_size(15)

            self.abort_button = Button(self, x=140, y=570, text='Abort', func=self.abort_generation)
            self.abort_button.height = 25
            self.abort_button.set_font_size(15)

            self.save_button = Button(self, x=260, y=570, text='Save', func=self.save)
            self.save_button.height = 25
            self.save_button.set_font_size(15)

            self.widgets = [
                self.oct_slider,
                self.sea_level_slider,
                self.draw_button,
                self.scale_slider,
                self.zoom_slider,
                self.width_slider,
                self.height_slider,
                self.pos_entry,
                self.lac_slider,
                self.seed_entry,
                self.clear_button,
                self.settings_button,
                self.abort_button,
                self.save_button
            ]

            self.width_chunks = 4
            self.height_chunks = 4

            for w in self.widgets:
                w.border_width = 1
                if isinstance(w, Slider):
                    w.pointer_border_width = 1

            self.movable_map_rect = pg.Rect((0, 0), (600, 600))
            self.dragging = False
            self.dist = vec(0, 0)

        def settings(self):
            start_settings.value = 1

        def save(self):
            filename = "image.png"
            surf = pg.transform.rotate(self.map, 90)
            surf = pg.transform.flip(surf, 0, 1)
            image = []
            for x in range(SETTINGS["mwd"]):
                row = []
                for y in range(SETTINGS["mhg"]):
                    row.append(surf.get_at((x, y)))
                image.append(row)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("saving image")
            cv2.imwrite(filename, image)

        def abort_generation(self):
            if generation_started.value:
                print("Generation aborted")
                for p in self.draw_processes:
                    p.terminate()
                    p.join()
                generation_started.value = 0

        def draw(self):
            SETTINGS["mwd"] = SETTINGS["wdh"]
            SETTINGS["mhg"] = SETTINGS["hgt"]

            SHARED_MAP.clear()
            chunks_generated.value = 0

            self.draw_processes = []

            for x in range(self.width_chunks):
                for y in range(self.height_chunks):
                    width = int(SETTINGS["wdh"]/self.width_chunks)
                    height = int(SETTINGS["hgt"]/self.height_chunks)
                    pos = [x*width, y*height]

                    self.draw_processes.append(Process(target=draw_colored, args=(SETTINGS["oct"], SETTINGS["scl"], SETTINGS["sea"], width,
                                                                                  height, SETTINGS["lac"], pos, SETTINGS["seed"],
                                                                                  SETTINGS["noise"], SETTINGS["offset"])))

            for p in self.draw_processes:
                p.start()

        def clear(self):
            self.map.fill((100, 100, 100))
            self.map_surface.fill(0)
            SHARED_MAP.clear()
            chunks_generated.value = 0

        def run(self):

            self.update()
            self.events()

        def reset_settings(self):
            self.oct_slider.set_mark(0)
            self.sea_level_slider.set_mark(60)
            self.scale_slider.set_mark(99)
            self.zoom_slider.set_mark(175)
            self.map.fill((100, 100, 100))
            self.width_slider.set_mark(50)
            self.height_slider.set_mark(50)
            self.lac_slider.set_mark(10)
            self.pos_entry.text = '0,0'
            self.seed_entry.text = '0'
            self.mmrect.topleft = self.map_rect.topleft
            self.movable_map_rect.topleft = (0, 0)
            self.dist = vec(0, 0)

        def update_settings(self):

            self.oct_slider.set_mark(SETTINGS["oct"]-1)
            self.sea_level_slider.set_mark(SETTINGS["sea"])
            self.scale_slider.set_mark(SETTINGS["scl"]-1)
            self.zoom_slider.set_mark(SETTINGS["zoom_slider"])
            self.zoom = SETTINGS["zoom"]
            self.width_slider.set_mark(int((SETTINGS["wdh"]-100)/10))
            self.height_slider.set_mark(int((SETTINGS["hgt"]-100)/10))
            pos = SETTINGS["offset"]
            x = str(pos[0])
            y = str(pos[1])
            self.pos_entry.text = x+','+y
            self.lac_slider.set_mark(SETTINGS["lac"]*10-10)
            self.seed_entry.text = str(SETTINGS["seed"])

        def update(self):
            mousepos = vec(pg.mouse.get_pos())
            p1 = pg.mouse.get_pressed()[0]
            # print(p1, mousepos)
            self.screen.fill((200, 200, 200))
            if reset_settings.value:
                reset_settings.value = 0
                self.reset_settings()

            if apply_settings.value:
                apply_settings.value = 0
                self.update_settings()

            if generation_finished.value:
                generation_finished.value = 0

                width = len(SHARED_MAP.values()[0])
                height = len(SHARED_MAP.values()[0][0])

                self.map = pg.Surface((width*self.width_chunks, height*self.height_chunks))
                self.map.fill((100, 100, 100))
                self.map_surface.fill((100, 100, 100))

                for c in SHARED_MAP.items():
                    pos = c[0]
                    chunk = c[1]
                    map_chunk = pg.pixelcopy.make_surface(chunk)
                    self.map.blit(map_chunk, (pos[1], pos[0]))

                self.map = pg.transform.flip(pg.transform.rotate(self.map, -90), 1, 0)

            self.movable_map = pg.transform.scale(
                self.map, (int(SETTINGS["mwd"]*self.zoom), int(SETTINGS["mhg"]*self.zoom)))
            self.movable_map_rect.size = self.movable_map.get_rect().size
            self.mmrect = self.movable_map_rect.copy()
            self.mmrect.topleft = vec(self.mmrect.topleft) + vec(575, 25)

            if self.mmrect.collidepoint(mousepos) and self.map_rect.collidepoint(mousepos):
                mpos = vec(mousepos) - vec(575, 25)

                if p1 and not self.dragging:
                    self.dragging = True
                    self.dist = mousepos - vec(self.mmrect.topleft)

                elif self.dragging and p1:
                    self.mmrect.topleft = mousepos - self.dist
                    self.movable_map_rect.center = vec(self.mmrect.center) - vec(575, 25)
                elif self.dragging and not p1:
                    self.dragging = False

            self.map_surface.fill(0)
            self.map_surface.blit(self.movable_map, self.movable_map_rect)
            self.screen.blit(self.map_surface, self.map_rect)
            pg.draw.rect(self.screen, (0, 0, 0), self.map_rect, 1)
            for w in self.widgets:
                w.update()

            SETTINGS["oct"] = self.oct_slider.mark+1
            SETTINGS["scl"] = self.scale_slider.mark+1
            SETTINGS["sea"] = self.sea_level_slider.mark
            try:
                seed = int(self.seed_entry.text)
            except:
                if self.seed_entry.text.strip() != '':
                    seed = int("".join([str(ord(c)) for c in self.seed_entry.text]))
                else:
                    seed = 0

            self.oct_slider.set_label("Octaves: "+str(self.oct_slider.mark+1))
            self.sea_level_slider.set_label("Sea level: "+str(self.sea_level_slider.mark))
            self.scale_slider.set_label("Scale: "+str(self.scale_slider.mark+1))

            self.zoom_slider.set_label("Zoom: "+str(self.zoom_slider.max-self.zoom_slider.mark-25))
            self.zoom = (self.zoom_slider.max-self.zoom_slider.mark)/25
            SETTINGS["zoom_slider"] = self.zoom_slider.mark
            if self.zoom <= 0:
                self.zoom = 0.025

            SETTINGS["zoom"] = self.zoom
            # print(self.zoom)

            self.width_slider.set_label("Width: "+str((self.width_slider.mark)*10+100))
            SETTINGS["wdh"] = (self.width_slider.mark)*10+100
            self.height_slider.set_label("Height: "+str((self.height_slider.mark)*10+100))
            SETTINGS["hgt"] = (self.height_slider.mark)*10+100

            self.lac_slider.set_label("Lacunarity: "+str((self.lac_slider.mark+10)/10))
            SETTINGS["lac"] = (self.lac_slider.mark+10)/10

            pos = self.pos_entry.text.strip().split(',')
            for i, p in enumerate(pos):
                if p != '':
                    try:
                        pos[i] = int(p)
                    except:
                        pos[i] = 0

            SETTINGS["offset"] = pos

            pg.display.flip()

        def events(self):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    if not in_settings.value:
                        quit_settings.value = 1
                        raise SystemExit
                if event.type == pg.MOUSEWHEEL:
                    if self.map_rect.collidepoint(pg.mouse.get_pos()):
                        try:
                            self.zoom_slider.set_mark(self.zoom_slider.mark-event.y)
                        except:
                            self.zoom_slider.set_mark(0)

    e = Editor()
    while True:
        e.run()


if __name__ == '__main__':
    ed = Process(target=editor)
    set = Process(target=sw)
    set.start()
    ed.start()
    set.join()
    ed.join()
