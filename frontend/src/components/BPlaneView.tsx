import type { BPlaneData } from '../api/client';

interface Props {
    data: BPlaneData;
}

const SVG_SIZE = 300;
const CENTER = SVG_SIZE / 2;

export function BPlaneView({ data }: Props) {
    // Scale: map km to pixels, auto-scale based on largest ellipse
    const maxExtent = data.ellipses.length > 0
        ? Math.max(...data.ellipses.map(e => Math.max(e.semi_major_km, e.semi_minor_km))) * 3
        : Math.max(data.b_magnitude_km, 1);
    const scale = (SVG_SIZE * 0.4) / maxExtent;

    const bx = data.b_t_km * scale;
    const by = -data.b_r_km * scale; // invert Y for SVG
    const hbr = data.hard_body_radius_km * scale;

    const ELLIPSE_COLORS = ['rgba(59, 130, 246, 0.5)', 'rgba(59, 130, 246, 0.3)', 'rgba(59, 130, 246, 0.15)'];

    return (
        <div className="card" style={{ padding: 16 }}>
            <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                B-Plane Conjunction Geometry
            </div>

            <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
                <svg width={SVG_SIZE} height={SVG_SIZE} style={{ background: 'var(--bg)', borderRadius: 6, border: '1px solid var(--brd)' }}>
                    {/* Axes */}
                    <line x1={CENTER} y1={10} x2={CENTER} y2={SVG_SIZE - 10} stroke="var(--brd)" strokeWidth={1} />
                    <line x1={10} y1={CENTER} x2={SVG_SIZE - 10} y2={CENTER} stroke="var(--brd)" strokeWidth={1} />
                    <text x={SVG_SIZE - 20} y={CENTER - 6} fill="var(--t4)" fontSize={10} textAnchor="middle">B·T</text>
                    <text x={CENTER + 10} y={18} fill="var(--t4)" fontSize={10} textAnchor="start">B·R</text>

                    {/* Covariance ellipses (3σ → 1σ order for layering) */}
                    {[...data.ellipses].reverse().map((e, i) => (
                        <ellipse
                            key={e.sigma}
                            cx={CENTER}
                            cy={CENTER}
                            rx={e.semi_major_km * scale * e.sigma}
                            ry={e.semi_minor_km * scale * e.sigma}
                            transform={`rotate(${e.rotation_deg} ${CENTER} ${CENTER})`}
                            fill={ELLIPSE_COLORS[data.ellipses.length - 1 - i] ?? 'rgba(59,130,246,0.1)'}
                            stroke="rgba(59, 130, 246, 0.6)"
                            strokeWidth={0.5}
                        />
                    ))}

                    {/* Hard-body radius circle */}
                    <circle cx={CENTER} cy={CENTER} r={Math.max(hbr, 2)} fill="rgba(239, 68, 68, 0.2)" stroke="#EF4444" strokeWidth={1} />

                    {/* Miss vector arrow */}
                    <line x1={CENTER} y1={CENTER} x2={CENTER + bx} y2={CENTER + by} stroke="#F59E0B" strokeWidth={1.5} markerEnd="url(#arrow)" />
                    <defs>
                        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" fill="#F59E0B" />
                        </marker>
                    </defs>

                    {/* Miss point */}
                    <circle cx={CENTER + bx} cy={CENTER + by} r={3} fill="#F59E0B" />

                    {/* Labels */}
                    {data.ellipses.map(e => (
                        <text key={`l${e.sigma}`} x={CENTER + e.semi_major_km * scale * e.sigma + 4} y={CENTER - 4} fill="var(--t4)" fontSize={8}>
                            {e.sigma}σ
                        </text>
                    ))}
                </svg>

                <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {[
                        { l: 'B·T', v: `${data.b_t_km.toFixed(4)} km` },
                        { l: 'B·R', v: `${data.b_r_km.toFixed(4)} km` },
                        { l: '|B|', v: `${data.b_magnitude_km.toFixed(4)} km` },
                        { l: 'Miss Distance', v: `${data.miss_distance_km.toFixed(4)} km` },
                        { l: 'Hard Body', v: `${(data.hard_body_radius_km * 1000).toFixed(1)} m` },
                        { l: 'B-Plane Pc', v: data.collision_probability_bplane.toExponential(4) },
                        { l: 'σ_T', v: `${data.sigma_t_km.toFixed(4)} km` },
                        { l: 'σ_R', v: `${data.sigma_r_km.toFixed(4)} km` },
                        { l: 'Encounter ∠', v: `${data.encounter_angle_deg.toFixed(1)}°` },
                        { l: 'V_rel ∠', v: `${data.relative_velocity_angle_deg.toFixed(1)}°` },
                        { l: 'V_rel', v: `${data.relative_velocity_km_s.toFixed(3)} km/s` },
                    ].map(p => (
                        <div key={p.l} style={{ padding: '6px 8px', borderRadius: 4, background: 'var(--bg-h)' }}>
                            <div style={{ fontSize: '.6rem', color: 'var(--t4)' }}>{p.l}</div>
                            <div className="font-mono" style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t1)' }}>{p.v}</div>
                        </div>
                    ))}
                </div>
            </div>

            {data.ellipses.length > 0 && (
                <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
                    {data.ellipses.map(e => (
                        <div key={e.sigma} style={{ flex: 1, padding: '6px 8px', borderRadius: 4, background: 'var(--bg-h)', textAlign: 'center' }}>
                            <div style={{ fontSize: '.6rem', color: 'var(--t4)' }}>{e.sigma}σ Ellipse</div>
                            <div className="font-mono" style={{ fontSize: '.65rem', color: 'var(--t2)' }}>
                                {e.semi_major_km.toFixed(4)} × {e.semi_minor_km.toFixed(4)} km
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
