import puppeteer from 'puppeteer';

(async () => {
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    
    // Set viewport
    await page.setViewport({ width: 1280, height: 800 });

    // Capture console
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.message));

    console.log('Navigating to http://localhost:5174...');
    await page.goto('http://localhost:5174', { waitUntil: 'networkidle2' });
    
    console.log('Waiting 15 seconds for React/Globe to render...');
    await new Promise(r => setTimeout(r, 15000));

    await page.screenshot({ path: 'globe_glitch.png' });
    console.log('Screenshot saved to globe_glitch.png');

    await browser.close();
})();