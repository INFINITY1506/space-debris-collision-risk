import { useState, useMemo, Fragment } from 'react';
import type { ThreatItem } from '../api/client';

interface Props {
    threats: ThreatItem[];
    satellite: { name: string; norad_id: number; altitude_km: number; inclination_deg: number };
    meta: { pairs: number; totalTime: number; inferTime: number };
}

type SK = 'rank' | 'collision_probability' | 'miss_distance_km' | 'relative_velocity_km_s' | 'uncertainty_pct';
type F = 'ALL' | 'HIGH' | 'MEDIUM' | 'LOW';

const BC: Record<string, string> = { HIGH: 'badge-high', MEDIUM: 'badge-medium', LOW: 'badge-low' };
const RC: Record<string, string> = { HIGH: 'var(--r-high)', MEDIUM: 'var(--r-med)', LOW: 'var(--r-low)' };
const RCraw: Record<string, string> = { HIGH: '#EF4444', MEDIUM: '#F59E0B', LOW: '#22C55E' };

function Badge({ level }: { level: string }) {
    return <span className={`${BC[level] ?? 'badge-low'} font-mono`} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 8px', borderRadius: 4, fontSize: '.68rem', fontWeight: 600 }}>
        <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'currentColor' }} />{level}
    </span>;
}

function tcaRel(utc: string) {
    try { const d = new Date(utc).getTime() - Date.now(); if (d < 0) return 'Past'; const h = Math.floor(d / 36e5); return h >= 24 ? `in ${Math.floor(h / 24)}d ${h % 24}h` : h > 0 ? `in ${h}h` : 'in <1h'; } catch { return ''; }
}

export function ResultsTable({ threats, satellite, meta }: Props) {
    const [sk, setSk] = useState<SK>('rank');
    const [desc, setDesc] = useState(false);
    const [filt, setFilt] = useState<F>('ALL');
    const [exp, setExp] = useState<number | null>(null);

    const sort = (k: SK) => { if (sk === k) setDesc(d => !d); else { setSk(k); setDesc(k !== 'rank'); } };

    const rows = useMemo(() => {
        const f = filt === 'ALL' ? threats : threats.filter(t => t.risk_level === filt);
        return [...f].sort((a, b) => { const d = (a[sk] as number) - (b[sk] as number); return desc ? -d : d; });
    }, [threats, filt, sk, desc]);

    const cnt = useMemo(() => ({ HIGH: threats.filter(t => t.risk_level === 'HIGH').length, MEDIUM: threats.filter(t => t.risk_level === 'MEDIUM').length, LOW: threats.filter(t => t.risk_level === 'LOW').length }), [threats]);

    const Arrow = ({ k }: { k: SK }) => sk !== k
        ? <svg width="8" height="8" viewBox="0 0 10 10" fill="currentColor" style={{ opacity: .25, marginLeft: 3, display: 'inline' }}><path d="M5 1l3 4H2l3-4zm0 8L2 5h6L5 9z" /></svg>
        : <svg width="8" height="8" viewBox="0 0 10 10" fill="var(--accent)" style={{ marginLeft: 3, display: 'inline' }}><path d={desc ? "M5 9L2 5h6L5 9z" : "M5 1l3 4H2L5 1z"} /></svg>;

    return (
        <div className="anim-fade-2" style={{ overflow: 'hidden', background: 'rgba(255,255,255,.02)', borderRadius: 8, border: '1px solid var(--glass-border)' }}>
            <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--glass-border)' }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                    <div>
                        <div style={{ fontSize: '.78rem', fontWeight: 600, color: 'var(--t1)' }}>Threat Analysis</div>
                        <div style={{ fontSize: '.65rem', color: 'var(--t4)' }}>{threats.length} threats &middot; {satellite.name}</div>
                    </div>
                    <div style={{ display: 'flex', gap: 3 }}>
                        {(['ALL', 'HIGH', 'MEDIUM', 'LOW'] as F[]).map(f => {
                            const n = f === 'ALL' ? threats.length : cnt[f];
                            const on = filt === f;
                            return <button key={f} onClick={() => setFilt(f)}
                                className={f !== 'ALL' ? BC[f] : ''}
                                style={{ padding: '3px 8px', borderRadius: 4, fontSize: '.65rem', fontWeight: 600, cursor: 'pointer', opacity: on ? 1 : .4, transition: 'opacity .15s',
                                    ...(f === 'ALL' ? { background: on ? 'var(--accent-m)' : 'transparent', color: on ? 'var(--accent)' : 'var(--t4)', border: '1px solid var(--brd)' } : {}) }}>
                                {f}{n > 0 ? ` (${n})` : ''}
                            </button>;
                        })}
                    </div>
                </div>
            </div>

            <div style={{ overflowX: 'auto' }}>
                <table className="tbl">
                    <thead><tr>
                        <th onClick={() => sort('rank')}># <Arrow k="rank" /></th>
                        <th>Object</th>
                        <th onClick={() => sort('collision_probability')}>Probability <Arrow k="collision_probability" /></th>
                        <th onClick={() => sort('uncertainty_pct')}>Uncertainty <Arrow k="uncertainty_pct" /></th>
                        <th>Risk</th>
                        <th onClick={() => sort('miss_distance_km')}>Miss Dist <Arrow k="miss_distance_km" /></th>
                        <th>TCA</th>
                        <th onClick={() => sort('relative_velocity_km_s')}>Velocity <Arrow k="relative_velocity_km_s" /></th>
                    </tr></thead>
                    <tbody>
                        {rows.map(t => {
                            const isExp = exp === t.rank;
                            const clr = RCraw[t.risk_level] ?? '#22C55E';
                            const pct = Math.min(100, t.collision_probability);
                            return (
                                <Fragment key={t.rank}>
                                <tr onClick={() => setExp(isExp ? null : t.rank)} style={{ cursor: 'pointer' }}>
                                    <td><span className="font-mono" style={{ fontSize: '.72rem', color: 'var(--t2)' }}>#{t.rank}</span></td>
                                    <td>
                                        <div style={{ fontWeight: 500, fontSize: '.82rem' }}>{t.debris_name}</div>
                                        <div className="font-mono" style={{ fontSize: '.65rem', color: 'var(--t4)' }}>NORAD {t.debris_norad_id}</div>
                                    </td>
                                    <td>
                                        <div className="font-mono" style={{ fontSize: '.8rem', fontWeight: 600, color: clr, marginBottom: 3 }}>
                                            {t.collision_probability < 1 ? `${(t.collision_probability).toFixed(4)}%` : `${t.collision_probability.toFixed(3)}%`}
                                        </div>
                                        <div className="prob-track"><div className="prob-fill" style={{ width: `${Math.max(pct, 1)}%`, background: clr }} /></div>
                                    </td>
                                    <td><span className="font-mono" style={{ fontSize: '.78rem', color: 'var(--t2)' }}>{t.uncertainty_range}</span></td>
                                    <td><Badge level={t.risk_level} /></td>
                                    <td><span className="font-mono" style={{ fontSize: '.8rem', fontWeight: 600 }}>
                                        {t.miss_distance_km < 1 ? `${(t.miss_distance_km * 1000).toFixed(0)} m` : `${t.miss_distance_km.toFixed(2)} km`}
                                    </span></td>
                                    <td>
                                        <div className="font-mono" style={{ fontSize: '.7rem', color: 'var(--t2)' }}>{t.tca_utc.replace('T', ' ').replace('Z', '')}</div>
                                        <div className="font-mono" style={{ fontSize: '.62rem', color: 'var(--accent)' }}>{tcaRel(t.tca_utc)}</div>
                                    </td>
                                    <td><span className="font-mono" style={{ fontSize: '.8rem', fontWeight: 600 }}>{t.relative_velocity_km_s.toFixed(2)} km/s</span></td>
                                </tr>
                                {isExp && <tr><td colSpan={8} style={{ padding: 0 }}>
                                    <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,.02)', borderTop: '1px solid var(--glass-border)', display: 'flex', flexDirection: 'column', gap: 8 }}>
                                        <div className="card" style={{ padding: 12, borderRadius: 6 }}>
                                            <div style={{ fontSize: '.62rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.05em' }}>Class Probabilities</div>
                                            {[{ l: 'LOW', v: t.probability_low, c: '#22C55E' }, { l: 'MED', v: t.probability_medium, c: '#F59E0B' }, { l: 'HIGH', v: t.probability_high, c: '#EF4444' }].map(p => (
                                                <div key={p.l} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                                                    <span className="font-mono" style={{ width: 32, fontSize: '.62rem', fontWeight: 600, color: p.c }}>{p.l}</span>
                                                    <div className="prob-track" style={{ flex: 1, height: 4 }}><div className="prob-fill" style={{ width: `${p.v * 100}%`, background: p.c }} /></div>
                                                    <span className="font-mono" style={{ width: 42, textAlign: 'right', fontSize: '.62rem', color: 'var(--t2)' }}>{(p.v * 100).toFixed(2)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="card" style={{ padding: 12, borderRadius: 6 }}>
                                            <div style={{ fontSize: '.62rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.05em' }}>Uncertainty</div>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 8px', borderRadius: 4, background: 'var(--bg-h)', marginBottom: 4 }}>
                                                <span style={{ fontSize: '.7rem', color: 'var(--t4)' }}>Total</span>
                                                <span className="font-mono" style={{ fontSize: '.7rem', fontWeight: 600 }}>{t.uncertainty_pct.toFixed(3)}%</span>
                                            </div>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 8px', borderRadius: 4, background: 'var(--bg-h)' }}>
                                                <span style={{ fontSize: '.7rem', color: 'var(--t4)' }}>Epistemic</span>
                                                <span className="font-mono" style={{ fontSize: '.7rem', fontWeight: 600 }}>{t.epistemic_uncertainty.toFixed(6)}</span>
                                            </div>
                                        </div>
                                        <div className="card" style={{ padding: 12, borderRadius: 6 }}>
                                            <div style={{ fontSize: '.62rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.05em' }}>Encounter</div>
                                            {[{ l: 'Miss Dist', v: `${t.miss_distance_km.toFixed(3)} km` }, { l: 'Velocity', v: `${t.relative_velocity_km_s.toFixed(3)} km/s` }, { l: 'TCA', v: t.tca_utc }].map(p => (
                                                <div key={p.l} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 8px', borderRadius: 4, background: 'var(--bg-h)', marginBottom: 4 }}>
                                                    <span style={{ fontSize: '.7rem', color: 'var(--t4)' }}>{p.l}</span>
                                                    <span className="font-mono" style={{ fontSize: '.65rem', fontWeight: 600 }}>{p.v}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </td></tr>}
                                </Fragment>
                            );
                        })}
                    </tbody>
                </table>
            </div>
            {rows.length === 0 && <div style={{ padding: 40, textAlign: 'center', color: 'var(--t4)' }}>No threats matching filter.</div>}
        </div>
    );
}
