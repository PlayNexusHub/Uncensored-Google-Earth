/**
 * Google Earth Engine Script for NDVI Comparison
 * 
 * This script provides an alternative to the local Python processing workflow.
 * It can be run directly in the Google Earth Engine Code Editor.
 * 
 * Features:
 * - Load Sentinel-2 imagery for a user-defined region
 * - Compute NDVI for two different dates
 * - Perform change detection analysis
 * - Visualize results with interactive controls
 */

// Define the region of interest (modify these coordinates as needed)
var roi = ee.Geometry.Rectangle([-122.6, 37.6, -122.3, 37.9]); // San Francisco Bay Area example

// Define the two dates for comparison
var date1 = '2023-07-01';
var date2 = '2023-07-25';

// Function to mask clouds using Sentinel-2 QA60 band
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 6;
  var cirrusBitMask = 1 << 7;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

// Function to compute NDVI
function computeNDVI(image) {
  var nir = image.select('B8');
  var red = image.select('B4');
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  return ndvi;
}

// Load and process Sentinel-2 imagery for date1
var s2_collection1 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate(date1, ee.Date(date1).advance(1, 'day'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds);

var s2_image1 = s2_collection1.first();
var ndvi1 = computeNDVI(s2_image1);

// Load and process Sentinel-2 imagery for date2
var s2_collection2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate(date2, ee.Date(date2).advance(1, 'day'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds);

var s2_image2 = s2_collection2.first();
var ndvi2 = computeNDVI(s2_image2);

// Compute change detection
var ndvi_change = ndvi2.subtract(ndvi1).rename('NDVI_Change');
var ndvi_abs_change = ndvi_change.abs().rename('NDVI_Absolute_Change');

// Create RGB composite for date1 (using B4, B3, B2 for true color)
var rgb1 = s2_image1.select(['B4', 'B3', 'B2']).rename(['red', 'green', 'blue']);

// Create RGB composite for date2
var rgb2 = s2_image2.select(['B4', 'B3', 'B2']).rename(['red', 'green', 'blue']);

// Set visualization parameters
var ndviParams = {
  min: -1,
  max: 1,
  palette: ['red', 'yellow', 'green']
};

var changeParams = {
  min: -0.5,
  max: 0.5,
  palette: ['red', 'white', 'green']
};

var rgbParams = {
  min: 0,
  max: 0.3
};

// Add layers to the map
Map.centerObject(roi, 10);

// Add RGB composites
Map.addLayer(rgb1, rgbParams, 'True Color - Date 1', false);
Map.addLayer(rgb2, rgbParams, 'True Color - Date 2', false);

// Add NDVI layers
Map.addLayer(ndvi1, ndviParams, 'NDVI - Date 1');
Map.addLayer(ndvi2, ndviParams, 'NDVI - Date 2');

// Add change detection layers
Map.addLayer(ndvi_change, changeParams, 'NDVI Change (Date2 - Date1)');
Map.addLayer(ndvi_abs_change, {min: 0, max: 0.5, palette: ['white', 'red']}, 'NDVI Absolute Change');

// Add the ROI outline
Map.addLayer(roi, {color: 'blue'}, 'Region of Interest');

// Create a panel with information
var panel = ui.Panel({
  style: {
    width: '300px'
  }
});

panel.add(ui.Label('NDVI Comparison Tool', {fontSize: '16px', fontWeight: 'bold'}));
panel.add(ui.Label('Date 1: ' + date1));
panel.add(ui.Label('Date 2: ' + date2));
panel.add(ui.Label(''));
panel.add(ui.Label('Instructions:'));
panel.add(ui.Label('• Toggle layers on/off using the Layers panel'));
panel.add(ui.Label('• Use the NDVI Change layer to see vegetation changes'));
panel.add(ui.Label('• Red areas = decreased vegetation'));
panel.add(ui.Label('• Green areas = increased vegetation'));

Map.add(panel);

// Export results (optional - uncomment to download)
// Export.image.toDrive({
//   image: ndvi1,
//   description: 'NDVI_Date1',
//   folder: 'GEE_NDVI_Comparison',
//   region: roi,
//   scale: 10
// });

// Export.image.toDrive({
//   image: ndvi2,
//   description: 'NDVI_Date2',
//   folder: 'GEE_NDVI_Comparison',
//   region: roi,
//   scale: 10
// });

// Export.image.toDrive({
//   image: ndvi_change,
//   description: 'NDVI_Change',
//   folder: 'GEE_NDVI_Comparison',
//   region: roi,
//   scale: 10
// });

print('NDVI Comparison Complete!');
print('Check the Layers panel to toggle different visualizations.');
