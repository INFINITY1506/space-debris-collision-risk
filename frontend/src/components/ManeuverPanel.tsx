import type { ManeuverResponse } from '../api/client';

interface Props {
    data: ManeuverResponse;
}

const EFF_COLORS: Record<string, string> = { high: '#22C55E', medium: '#F59E0B', low: '#EF4444' };
const TYPE_COLORS: Record<string, string> = { 'last-orbit': '#EF4444', 'penultimate-orbit': '#F59E0B', 'early-warning': '#22C55E' };

export function ManeuverPanel({ data }: Props) {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {/* Header stats */}
            <div className="card" style={{ padding: 14 }}>
                <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                    Avoidance Maneuver Analysis
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
                    {[
                        { l: 'Time to TCA', v: `${data.time_to_tca_hours.toFixed(1)}h` },
                        { l: 'Current Miss', v: `${data.current_miss_distance_km.toFixed(3)} km` },
                        { l: 'Target Miss', v: `${data.target_miss_distance_km.toFixed(1)} km` },
                        { l: 'Action', v: data.recommended_action, c: data.recommended_action === 'EXECUTE_MANEUVER' ? '#EF4444' : data.recommended_action === 'PLAN_MANEUVER' ? '#F59E0B' : '#22C55E' },
                    ].map(s => (
                        <div key={s.l} style={{ padding: '8px 10px', borderRadius: 4, background: 'var(--bg-h)' }}>
                            <div style={{ fontSize: '.6rem', color: 'var(--t4)' }}>{s.l}</div>
                            <div className="font-mono" style={{ fontSize: '.78rem', fontWeight: 600, color: s.c ?? 'var(--t1)' }}>{s.v}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Maneuver options */}
            <div className="card" style={{ padding: 14 }}>
                <div style={{ fontSize: '.68rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                    Delta-V Options (RSW Frame)
                </div>
                <table className="tbl" style={{ fontSize: '.72rem' }}>
                    <thead>
                        <tr>
                            <th>Direction</th>
                            <th>ΔV (m/s)</th>
                            <th>R</th>
                            <th>S</th>
                            <th>W</th>
                            <th>Fuel Cost</th>
                            <th>Efficiency</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.maneuvers.map(m => (
                            <tr key={m.direction} style={m.recommended ? { background: 'rgba(59, 130, 246, 0.06)' } : undefined}>
                                <td>
                                    <div style={{ fontWeight: 500 }}>{m.direction_label}</div>
                                </td>
                                <td>
                                    <span className="font-mono" style={{ fontWeight: 600, color: 'var(--t1)' }}>
                                        {m.delta_v_m_s.toFixed(4)}
                                    </span>
                                </td>
                                <td className="font-mono">{m.delta_v_components.R.toFixed(2)}</td>
                                <td className="font-mono">{m.delta_v_components.S.toFixed(2)}</td>
                                <td className="font-mono">{m.delta_v_components.W.toFixed(2)}</td>
                                <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                        <div className="prob-track" style={{ width: 40, height: 4 }}>
                                            <div className="prob-fill" style={{ width: `${m.fuel_cost_relative * 100}%`, background: 'var(--accent)' }} />
                                        </div>
                                        <span className="font-mono" style={{ fontSize: '.65rem', color: 'var(--t2)' }}>{(m.fuel_cost_relative * 100).toFixed(0)}%</span>
                                    </div>
                                </td>
                                <td>
                                    <span style={{ color: EFF_COLORS[m.efficiency] ?? 'var(--t2)', fontWeight: 600, fontSize: '.68rem' }}>
                                        {m.efficiency.toUpperCase()}
                                    </span>
                                </td>
                                <td>
                                    {m.recommended && (
                                        <span style={{ padding: '2px 6px', borderRadius: 3, background: 'rgba(59,130,246,0.15)', color: 'var(--accent)', fontSize: '.6rem', fontWeight: 600 }}>
                                            RECOMMENDED
                                        </span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Maneuver windows timeline */}
            {data.maneuver_windows.length > 0 && (
                <div className="card" style={{ padding: 14 }}>
                    <div style={{ fontSize: '.68rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                        Maneuver Windows
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                        {data.maneuver_windows.map((w, i) => (
                            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 10px', borderRadius: 4, background: 'var(--bg-h)' }}>
                                <div style={{ width: 8, height: 8, borderRadius: '50%', background: TYPE_COLORS[w.type] ?? 'var(--t4)', flexShrink: 0 }} />
                                <div style={{ flex: 1 }}>
                                    <div className="font-mono" style={{ fontSize: '.68rem', color: 'var(--t2)' }}>
                                        {w.window_start_utc.replace('T', ' ').replace('Z', '')} — {w.window_end_utc.replace('T', ' ').replace('Z', '')}
                                    </div>
                                    <div style={{ fontSize: '.6rem', color: 'var(--t4)', marginTop: 1 }}>
                                        Optimal: {w.optimal_time_utc.replace('T', ' ').replace('Z', '')}
                                    </div>
                                </div>
                                <div style={{ textAlign: 'right' }}>
                                    <div className="font-mono" style={{ fontSize: '.7rem', fontWeight: 600, color: 'var(--t1)' }}>
                                        -{w.hours_before_tca.toFixed(1)}h
                                    </div>
                                    <div style={{ fontSize: '.58rem', color: TYPE_COLORS[w.type] ?? 'var(--t4)', fontWeight: 600, textTransform: 'uppercase' }}>
                                        {w.type.replace('-', ' ')}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
