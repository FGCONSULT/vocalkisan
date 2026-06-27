// /var/www/viksitdairy2047/server.js
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const regression = require('regression');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8000;
const DOMAIN = process.env.DOMAIN || 'https://viksitdairy2047.in/forecast2050/';

app.use(bodyParser.json({ limit: '1mb' }));
app.use(cors({ origin: [DOMAIN] }));

// Serve static frontend if needed
const staticDir = path.join(__dirname, 'static');
app.use('/', express.static(staticDir));

// Health check
app.get('/app/status', (req, res) => {
  res.json({ status: 'ok' });
});

// Forecast endpoint
app.post('/app/forecast', (req, res) => {
  try {
    const payload = req.body || {};
    const historical = payload.historical || [];
    const model = (payload.model || 'linear').toLowerCase();
    const targetYear = (payload.options && payload.options.target_year) || 2050;

    if (!Array.isArray(historical) || historical.length === 0) {
      return res.status(400).json({ error: 'historical data required' });
    }

    const data = historical
      .map(r => ({ year: Number(r.year), supply: Number(r.supply) }))
      .filter(r => Number.isFinite(r.year) && Number.isFinite(r.supply))
      .sort((a,b) => a.year - b.year);

    if (data.length < 2) {
      return res.status(400).json({ error: 'need at least two historical points' });
    }

    const points = data.map(d => [d.year, d.supply]);

    let result;
    if (model === 'poly2') {
      result = regression.polynomial(points, { order: 2, precision: 6 });
    } else if (model === 'poly3') {
      result = regression.polynomial(points, { order: 3, precision: 6 });
    } else {
      result = regression.linear(points, { precision: 6 });
    }

    const startYear = data[0].year;
    const years = [];
    const forecast = [];
    const histMap = new Map(data.map(d => [d.year, d.supply]));

    for (let y = startYear; y <= targetYear; y++) {
      years.push(y);
      const pred = result.predict(y)[1];
      forecast.push(Math.max(Number(pred.toFixed(2)), 0));
    }

    const historicalSeries = years.map(y => histMap.has(y) ? Number(histMap.get(y)) : null);

    const residuals = data.map(d => d.supply - result.predict(d.year)[1]);
    const sigma = residuals.length > 1 ? Math.sqrt(residuals.reduce((s,r)=>s+r*r,0)/(residuals.length-1)) : 0;
    const lower = forecast.map(v => Math.max(Number((v - 1.96 * sigma).toFixed(2)), 0));
    const upper = forecast.map(v => Number((v + 1.96 * sigma).toFixed(2)));

    res.json({
      years,
      historical: historicalSeries,
      forecast,
      intervals: { lower, upper },
      model,
      meta: { modelSummary: result.string, sigma: Number(sigma.toFixed(4)) }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'internal server error' });
  }
});

app.listen(PORT, () => {
  console.log(`Supply forecast API listening on port ${PORT}`);
});
