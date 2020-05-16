# Worley Noise

This project includes a serial and a parallel implementation of the Worley Noise on GPU. The parallel version uses CUDA.

## What is Worley noise?
Worley noise, sometimes referred to as cell noise, is used to add details to the surface of rendered objects [[1]](http://weber.itn.liu.se/~stegu/TNM084-2017/worley-originalpaper.pdf), a process known as procedural texturing. It can be used to create textures resembling stone, water, ice, intestines, crumpled paper, mountain ranges, cell noise,  crocodile skin and clouds.

## How is Worley noise computed?

The basic principle of the Worley noise algorithm initially involves taking random fixed points in a space. This project only focuses on 2-dimensional space, although 3-dimensional space is also possible. Then, for every point, 
 <img src="https://render.githubusercontent.com/render/math?math=(x, y)">,  in the <img src="https://render.githubusercontent.com/render/math?math=xy"> plane, the distance to the <img src="https://render.githubusercontent.com/render/math?math=n">th closest fixed point is calculated, <img src="https://render.githubusercontent.com/render/math?math=F_n(x, y)">. The distance is then normalized to create a shade of grey. The figures below show how 4 random points are transformed into Worley noise. The tiling (red boxed) are used as an optimizations as suggested by Rosén [[2]](http://www.carljohanrosen.com/share/CellNoiseAndProcessing.pdf).

Random points | Noise from random points
:-------------------------:|:-------------------------:
![1 pixel per tile](examples/3x2.png)  |  ![5 pixels per tile](examples/3x2points.png)


# Example outputs

1 Pixel/Tile | 5 Pixels/Tile
:-------------------------:|:-------------------------:
![1 pixel per tile](examples/PPtile1.jpeg)  |  ![5 pixels per tile](examples/PPtile5.jpeg)

Intensity = 1 | Intensity = 1.5
:-------------------------:|:-------------------------:
![intensity 1](examples/Intensity1.jpeg) | ![intensity 1.5](examples/Intensity1.5.jpeg)

Normal | Reverse
:-------------------------:|:-------------------------:
![normal noise](examples/NoReverse.jpeg) | ![reverse noise](examples/Reverse.jpeg)
