import type { SatelliteInfo } from '../api/client';

export function SatelliteCard({ satellite: s, time, pairs }: { satellite: SatelliteInfo; time: number; pairs: number }) {
    const type = s.altitude_km < 2000 ? 'LEO' : s.altitude_km > 35286 && s.altitude_km < 36286 ? 'GEO' : 'MEO';

    return (
        <div className="card anim-fade" style={{ padding: '10px 12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: '.82rem', fontWeight: 600, color: 'var(--t1)' }}>{s.name}</span>
                <span className="font-mono" style={{ fontSize: '.58rem', fontWeight: 600, padding: '1px 5px', borderRadius: 3, color: 'var(--t2)', background: 'var(--accent-m)', border: '1px solid var(--brd)' }}>{type}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '4px 12px', fontSize: '.72rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--t4)', fontSize: '.6rem', textTransform: 'uppercase' }}>NORAD</span>
                    <span className="font-mono" style={{ fontWeight: 600 }}>{s.norad_id}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--t4)', fontSize: '.6rem', textTransform: 'uppercase' }}>ALT</span>
                    <span className="font-mono" style={{ fontWeight: 600 }}>{s.altitude_km.toFixed(1)} km</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--t4)', fontSize: '.6rem', textTransform: 'uppercase' }}>INC</span>
                    <span className="font-mono" style={{ fontWeight: 600 }}>{s.inclination_deg.toFixed(2)}&deg;</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--t4)', fontSize: '.6rem', textTransform: 'uppercase' }}>ECC</span>
                    <span className="font-mono" style={{ fontWeight: 600 }}>{s.eccentricity.toFixed(6)}</span>
                </div>
            </div>
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                <div style={{ flex: 1, textAlign: 'center', padding: '5px 8px', borderRadius: 4, background: 'rgba(255,255,255,.03)', border: '1px solid var(--glass-border)' }}>
                    <div className="font-mono" style={{ fontSize: '.82rem', fontWeight: 600 }}>{pairs.toLocaleString()}</div>
                    <div style={{ fontSize: '.55rem', color: 'var(--t4)' }}>Pairs</div>
                </div>
                <div style={{ flex: 1, textAlign: 'center', padding: '5px 8px', borderRadius: 4, background: 'rgba(255,255,255,.03)', border: '1px solid var(--glass-border)' }}>
                    <div className="font-mono" style={{ fontSize: '.82rem', fontWeight: 600 }}>{time.toFixed(1)}s</div>
                    <div style={{ fontSize: '.55rem', color: 'var(--t4)' }}>Time</div>
                </div>
            </div>
        </div>
    );
}
