/**
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
 */

function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

var area= ee.Geometry.Rectangle(-120.25, 36.45, -119.65,37.05) //Extracted Area


// Map the function over one year of data and take the median.
// Load Sentinel-2 TOA reflectance data.
var dataset = ee.ImageCollection('COPERNICUS/S2')
                  .filterDate('2016-01-01', '2016-12-31')
                  .filterBounds(area)
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskS2clouds);

var mosaic = dataset.select(['B4', 'B3', 'B2']).mosaic()

var rgbVis = {
  min: 0.0,
  max: 0.3,
  bands: ['B4', 'B3', 'B2'],
};

Map.setCenter(-119.71, 36.45, 10);
Map.addLayer(mosaic, rgbVis, 'RGB');

var Sentinel_RGB = mosaic.visualize({bands: ['B4', 'B3', 'B2'], min: 0.0, max: 0.3});

Export.image.toDrive({
  image: Sentinel_RGB,
  description: 'Sentinel_image',
  scale: 10,
  region: area, // .geometry().bounds() needed for multipolygon
  maxPixels: 2000000000
});
