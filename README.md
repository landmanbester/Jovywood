# Jovywood

![GitHub Logo](/images/jupiter_glasses.jpeg)

## List of workers
Currently available workers (listed in the order you would typically run them in)

* concat - concatenates a series of fits files and writes them out as a chunked zarr array.
 All subsequent workers accept chunked zarr arrays as input.

* extract - extracts a secion of the image specified by a time, x and y range (in pixels).
 This is an optional step that can be used to run the algo on a smaller section of the image.

* fit - fits hyper-parameters for all pixels along time (the most computationally expensive part).
 Writes out a zarr array containing hyper-parameters for each pixel.

* interp - interpolates the data using hyper-parameters produced by fit.
 Produces a zarr array with the time axis uniformly interpolated over the time range.
 An option oversmooth parameter can be used to artificially increase the degree of smoothing.

* cube2fits - converts the cube produced by interp into separate fits files with an additional time axis.

Run

$ smoovie --help

for a list of workers and

$ smoovie worker --help

for options pertaining to individual workers.