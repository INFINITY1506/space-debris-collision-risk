import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { MonteCarloResult } from '../../api/client';

interface Props {
    data: MonteCarloResult;
}

function buildHistogram(samples: number[], nBins: number = 20) {
    if (samples.length === 0) return [];
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    const range = max - min || 1e-15;
    const binWidth = range / nBins;

    const bins = Array.from({ length: nBins }, (_, i) => ({
        binStart: min + i * binWidth,
        binEnd: min + (i + 1) * binWidth,
        label: (min + (i + 0.5) * binWidth).toExponential(1),
        count: 0,
    }));

    for (const s of samples) {
        const idx = Math.min(Math.floor((s - min) / binWidth), nBins - 1);
        bins[idx].count++;
    }
    return bins;
}

export function MonteCarloHist({ data }: Props) {
    const bins = buildHistogram(data.samples, 15);

    return (
        <div className="card" style={{ padding: 16 }}>
            <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                Monte Carlo Distribution ({data.n_samples} samples)
            </div>
            <div style={{ display: 'flex', gap: 16, marginBottom: 10 }}>
                {[
                    { l: 'Mean', v: data.mean_probability.toExponential(3) },
                    { l: 'Std', v: data.std_probability.toExponential(3) },
                    { l: 'P90', v: data.p90.toExponential(3) },
                    { l: 'P99', v: data.p99.toExponential(3) },
                ].map(s => (
                    <div key={s.l}>
                        <span style={{ fontSize: '.6rem', color: 'var(--t4)' }}>{s.l}: </span>
                        <span className="font-mono" style={{ fontSize: '.65rem', color: 'var(--t2)' }}>{s.v}</span>
                    </div>
                ))}
            </div>
            <ResponsiveContainer width="100%" height={200}>
                <BarChart data={bins} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--brd)" />
                    <XAxis
                        dataKey="label"
                        tick={{ fontSize: 8, fill: 'var(--t4)' }}
                        stroke="var(--brd)"
                        interval={2}
                    />
                    <YAxis
                        tick={{ fontSize: 10, fill: 'var(--t4)' }}
                        stroke="var(--brd)"
                        width={30}
                        label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: 'var(--t4)' } }}
                    />
                    <Tooltip
                        contentStyle={{ background: 'var(--bg-s)', border: '1px solid var(--brd)', borderRadius: 6, fontSize: '0.7rem' }}
                        formatter={(v: any) => [v, 'Samples']}
                        labelFormatter={(l: any) => `Pc ~ ${l}`}
                    />
                    <Bar dataKey="count" fill="#3B82F6" radius={[2, 2, 0, 0]} />
                    {data.mean_probability > 0 && (
                        <ReferenceLine x={data.mean_probability.toExponential(1)} stroke="var(--r-high)" strokeDasharray="5 3" label={{ value: 'Mean', position: 'top', style: { fontSize: 9, fill: 'var(--r-high)' } }} />
                    )}
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}
