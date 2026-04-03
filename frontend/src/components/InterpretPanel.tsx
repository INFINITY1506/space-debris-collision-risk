import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { InterpretResponse } from '../api/client';

interface Props {
    data: InterpretResponse;
}

const CLASS_COLORS = ['#22C55E', '#F59E0B', '#EF4444'];
const CLASS_LABELS = ['LOW', 'MED', 'HIGH'];

export function InterpretPanel({ data }: Props) {
    const { attention, feature_importance, ensemble } = data;

    // Attention heatmap data: take first 48 hours for visibility
    const attnSlice = attention.cross_attention.slice(0, 48);
    const maxAttn = Math.max(...attnSlice, 1e-10);

    // Feature importance: top 15
    const topFeatures = feature_importance.slice(0, 15).map(f => ({
        name: f.feature.replace(/_/g, ' ').replace(/primary |secondary /g, (m) => m[0].toUpperCase() + m.slice(1)),
        importance: f.importance,
        gradient: f.raw_gradient,
    }));

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {/* Attention heatmap */}
            <div className="card" style={{ padding: 14 }}>
                <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                    Cross-Attention Weights
                </div>
                <div style={{ fontSize: '.6rem', color: 'var(--t4)', marginBottom: 8 }}>
                    Entropy: {attention.attention_entropy.toFixed(4)} | Peak: h{attention.peak_timesteps[0]?.hour ?? 0} ({((attention.peak_timesteps[0]?.weight ?? 0) * 100).toFixed(2)}%)
                </div>
                <div style={{ display: 'flex', gap: 1, height: 32, borderRadius: 4, overflow: 'hidden' }}>
                    {attnSlice.map((w, i) => {
                        const intensity = w / maxAttn;
                        return (
                            <div
                                key={i}
                                title={`Hour ${i}: ${(w * 100).toFixed(3)}%`}
                                style={{
                                    flex: 1,
                                    background: `rgba(59, 130, 246, ${0.05 + intensity * 0.9})`,
                                    minWidth: 1,
                                }}
                            />
                        );
                    })}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                    <span style={{ fontSize: '.55rem', color: 'var(--t4)' }}>0h</span>
                    <span style={{ fontSize: '.55rem', color: 'var(--t4)' }}>{attnSlice.length}h</span>
                </div>

                {/* Peak timesteps */}
                <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
                    {attention.peak_timesteps.slice(0, 6).map((p, i) => (
                        <div key={i} style={{ padding: '3px 8px', borderRadius: 3, background: i === 0 ? 'rgba(59,130,246,0.15)' : 'var(--bg-h)' }}>
                            <span className="font-mono" style={{ fontSize: '.62rem', color: i === 0 ? 'var(--accent)' : 'var(--t2)' }}>
                                h{p.hour} ({(p.weight * 100).toFixed(2)}%)
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Feature importance */}
            <div className="card" style={{ padding: 14 }}>
                <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                    Feature Importance (Gradient-Based)
                </div>
                <ResponsiveContainer width="100%" height={Math.max(200, topFeatures.length * 22)}>
                    <BarChart data={topFeatures} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 120 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--brd)" horizontal={false} />
                        <XAxis
                            type="number"
                            tick={{ fontSize: 9, fill: 'var(--t4)' }}
                            stroke="var(--brd)"
                            domain={[0, 1]}
                        />
                        <YAxis
                            type="category"
                            dataKey="name"
                            tick={{ fontSize: 9, fill: 'var(--t2)' }}
                            stroke="var(--brd)"
                            width={115}
                        />
                        <Tooltip
                            contentStyle={{ background: 'var(--bg-s)', border: '1px solid var(--brd)', borderRadius: 6, fontSize: '0.7rem' }}
                            formatter={(v: any) => [Number(v).toFixed(4), 'Importance']}
                        />
                        <Bar dataKey="importance" fill="#3B82F6" radius={[0, 3, 3, 0]} barSize={14} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Ensemble predictions */}
            <div className="card" style={{ padding: 14 }}>
                <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                    Ensemble Predictions ({ensemble.n_models} Models)
                </div>

                {/* Consensus bar */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 12 }}>
                    {[
                        { l: 'Agreement', v: `${(ensemble.agreement_score * 100).toFixed(1)}%` },
                        { l: 'Consensus', v: ensemble.consensus_class ?? '-' },
                        { l: 'Models', v: `${ensemble.n_models}` },
                    ].map(s => (
                        <div key={s.l} style={{ padding: '6px 8px', borderRadius: 4, background: 'var(--bg-h)', textAlign: 'center' }}>
                            <div style={{ fontSize: '.58rem', color: 'var(--t4)' }}>{s.l}</div>
                            <div className="font-mono" style={{ fontSize: '.78rem', fontWeight: 600, color: 'var(--t1)' }}>{s.v}</div>
                        </div>
                    ))}
                </div>

                {/* Mean probabilities */}
                <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: '.6rem', color: 'var(--t4)', marginBottom: 4 }}>Mean Class Probabilities</div>
                    {ensemble.mean_probabilities.map((p, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                            <span className="font-mono" style={{ width: 32, fontSize: '.62rem', fontWeight: 600, color: CLASS_COLORS[i] }}>{CLASS_LABELS[i]}</span>
                            <div className="prob-track" style={{ flex: 1, height: 6 }}>
                                <div className="prob-fill" style={{ width: `${p * 100}%`, background: CLASS_COLORS[i] }} />
                            </div>
                            <span className="font-mono" style={{ width: 48, textAlign: 'right', fontSize: '.62rem', color: 'var(--t2)' }}>
                                {(p * 100).toFixed(2)}%
                            </span>
                            {ensemble.std_probabilities[i] > 0 && (
                                <span className="font-mono" style={{ width: 48, textAlign: 'right', fontSize: '.55rem', color: 'var(--t4)' }}>
                                    ±{(ensemble.std_probabilities[i] * 100).toFixed(2)}
                                </span>
                            )}
                        </div>
                    ))}
                </div>

                {/* Individual model predictions */}
                {ensemble.individual_predictions.length > 1 && (
                    <div>
                        <div style={{ fontSize: '.6rem', color: 'var(--t4)', marginBottom: 4 }}>Individual Checkpoints</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                            {ensemble.individual_predictions.map((pred, i) => (
                                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px', borderRadius: 3, background: 'var(--bg-h)' }}>
                                    <span className="font-mono" style={{ fontSize: '.6rem', color: 'var(--t4)', width: 100, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {pred.checkpoint}
                                    </span>
                                    <div style={{ flex: 1, display: 'flex', gap: 4, height: 6 }}>
                                        {pred.probabilities.map((p, j) => (
                                            <div key={j} style={{ width: `${p * 100}%`, height: '100%', background: CLASS_COLORS[j], borderRadius: 1 }} />
                                        ))}
                                    </div>
                                    <span className="font-mono" style={{ fontSize: '.6rem', fontWeight: 600, color: CLASS_COLORS[['LOW', 'MEDIUM', 'HIGH'].indexOf(pred.predicted_class)] ?? 'var(--t2)' }}>
                                        {pred.predicted_class}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
