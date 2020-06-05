//var area= ee.Geometry.Rectangle(-119.71, 36.45, -119.65, 36.51)
var area = ee.Geometry.Rectangle(-120.25, 36.45, -119.65,37.05) //Tile2Vec Area

// Load four 2012 NAIP quarter quads, different locations.
var naip2016 = ee.ImageCollection('USDA/NAIP/DOQQ')
  .filterBounds(area)
  .filterDate('2016-01-01', '2016-12-31');

// Spatially mosaic the images in the collection and display.
var naip_mosaic = naip2016.mosaic();
print(naip_mosaic)

Map.setCenter(-119.71,36.45, 10);
Map.addLayer(naip_mosaic, {}, 'spatial mosaic');
Map.addLayer(area, {color: 'FF0000'}, 'geodesic polygon');

// Export a cloud-optimized GeoTIFF.
Export.image.toDrive({
  image: naip_mosaic,
  description: 'NAIPTile2VecArea',
  scale: 1,
  region: area,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});
