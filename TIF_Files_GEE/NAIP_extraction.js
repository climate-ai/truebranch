//var area= ee.Geometry.Rectangle(-119.71, 36.45, -119.65, 36.51)
var area = ee.Geometry.Rectangle(-120.25, 36.45, -119.65,37.05) //Extracted Area

// Load four 2012 NAIP quarter quads, different locations.
var naip2016 = ee.ImageCollection('USDA/NAIP/DOQQ')
  .filterBounds(area)
  .filterDate('2016-01-01', '2016-12-31');

// Spatially mosaic the images in the collection and display.
var naip_mosaic = naip2016.mosaic();

Map.setCenter(-119.71,36.45, 10);
Map.addLayer(naip_mosaic, {}, 'spatial mosaic');
Map.addLayer(area, {color: 'FF0000'}, 'geodesic polygon');

var Naip_RGB = naip_mosaic.visualize({bands: ['R', 'G', 'B']});

// Export a cloud-optimized GeoTIFF.
Export.image.toDrive({
  image: Naip_RGB,
  description: 'NAIP_image',
  scale: 1,
  region: area,
  maxPixels: 4464043320,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});
