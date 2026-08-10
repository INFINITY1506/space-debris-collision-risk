import { useState, useEffect } from 'react';

export function BottomBar() {
    const [utc, setUtc] = useState(new Date().toISOString().slice(11, 19));

    useEffect(() => {
        const t = setInterval(() => setUtc(new Date().toISOString().slice(11, 19)), 1000);
        return () => clearInterval(t);
    }, []);

    return (
        <div className="overlay-bottom glass bottom-bar">
            <div className="bottom-bar-legend">
                <div>
                    <i className="legend-dot legend-dot--active" />
                    <span className="font-mono">Active spacecraft</span>
                </div>
                <div>
                    <i className="legend-dot legend-dot--debris" />
                    <span className="font-mono">Debris object</span>
                </div>
            </div>
            <span className="bottom-bar-summary">CelesTrak TLE / SGP4 / Research screening only</span>
            <span className="font-mono bottom-bar-clock">UTC {utc}</span>
        </div>
    );
}
