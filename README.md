# Hackathon-wii-board
Use Wii Balance board to get weight distribution on a scanner. Use this to create a pressure map.

## [Presentation](https://docs.google.com/presentation/d/1NZb_9b_LQqPxPdZ_gRSTzicnDk_trhwMYQdjrNYcOCs/edit?slide=id.g391a2d3d94a_24_0#slide=id.g391a2d3d94a_24_0) 

<img width="1578" height="887" alt="image" src="https://github.com/user-attachments/assets/202f6074-857a-4483-8691-b1f395cf9b75" />

## Results: 

![loop_visualization-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/4b956dc7-37af-47c5-816d-86993792741d)

## Notes
- Got the weight measurements from each part of the wii board
- Downloaded a normal pressure map image and saved it as a template
- Manually annotated key regions of this template, which usually changes in color
- Each of these regions were assigned different weights for the 4 measurements from the wii board. i.e. The TL will affect left toe more than the right toe etc.
- For each mesurements all these were combined to a single image using the weights, some randomness etc.
- The image is filtered with multiple gaussian filters to make it look good. We don't display the foot when the weight on that side is too low
- Finally, this combined image was modified using the feet shape we get from the scanner results (we use one of a fixed scan for now)
