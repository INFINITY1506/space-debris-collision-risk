import { useState, useEffect } from 'react';

export function BottomBar() {
    const [utc, setUtc] = useState(new Date().toISOString().slice(11, 19));

    useEffect(() => {
        const t = setInterval(() => setUtc(new Date().toISOString().slice(11, 19)), 1000);
        return () => clearInterval(t);
    }, []);

    return (
        <div className="overlay-bottom glass" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 16px', fontSize: '.62rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--r-low)' }} />
                    <span className="font-mono" style={{ color: 'var(--t3)' }}>Active</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--r-high)' }} />
                    <span className="font-mono" style={{ color: 'var(--t3)' }}>Debris</span>
                </div>
            </div>
            <span style={{ color: 'var(--t4)' }}>Debris Sentinel v1.0 &middot; Transformer Model &middot; Research Use Only</span>
            <span className="font-mono" style={{ color: 'var(--t4)', letterSpacing: '.05em' }}>{utc} UTC</span>
        </div>
    );
}
