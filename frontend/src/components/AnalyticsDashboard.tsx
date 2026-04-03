import { useState, useEffect } from 'react';
import type {
    DetailedPredictResponse,
    ManeuverResponse,
    InterpretResponse,
    DetailedThreatItem,
} from '../api/client';
import { computeManeuver, interpretPrediction } from '../api/client';
import { TemporalRisk } from './charts/TemporalRisk';
import { ScatterPlot } from './charts/ScatterPlot';
import { UncertaintyChart } from './charts/UncertaintyChart';
import { MonteCarloHist } from './charts/MonteCarloHist';
import { BPlaneView } from './BPlaneView';
import { ManeuverPanel } from './ManeuverPanel';
import { InterpretPanel } from './InterpretPanel';

interface Props {
    detailed: DetailedPredictResponse;
}

type Tab = 'risk' | 'bplane' | 'maneuver' | 'interpret';

const TABS: { key: Tab; label: string }[] = [
    { key: 'risk', label: 'Risk Analysis' },
    { key: 'bplane', label: 'B-Plane' },
    { key: 'maneuver', label: 'Maneuver' },
    { key: 'interpret', label: 'Model Insight' },
];

export function AnalyticsDashboard({ detailed }: Props) {
    const [tab, setTab] = useState<Tab>('risk');
    const [selectedThreat, setSelectedThreat] = useState<number>(0);
    const [maneuverData, setManeuverData] = useState<ManeuverResponse | null>(null);
    const [interpretData, setInterpretData] = useState<InterpretResponse | null>(null);
    const [loading, setLoading] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const threats = detailed.threats;
    const currentThreat: DetailedThreatItem | undefined = threats[selectedThreat];

    // Lazy-load maneuver/interpret data when tab switches
    useEffect(() => {
        if (!currentThreat) return;

        if (tab === 'maneuver' && !maneuverData) {
            setLoading('Computing maneuver options...');
            setError(null);
            computeManeuver(
                detailed.satellite.name,
                undefined,
                currentThreat.debris_norad_id,
                5.0
            )
                .then(d => { setManeuverData(d); setLoading(null); })
                .catch(e => { setError(e.message); setLoading(null); });
        }

        if (tab === 'interpret' && !interpretData) {
            setLoading('Running interpretability analysis...');
            setError(null);
            interpretPrediction(
                detailed.satellite.name,
                undefined,
                currentThreat.debris_norad_id
            )
                .then(d => { setInterpretData(d); setLoading(null); })
                .catch(e => { setError(e.message); setLoading(null); });
        }
    }, [tab, currentThreat, maneuverData, interpretData, detailed.satellite.name]);

    // Reset lazy data when threat changes
    const handleThreatChange = (idx: number) => {
        setSelectedThreat(idx);
        setManeuverData(null);
        setInterpretData(null);
        setError(null);
    };

    return (
        <div className="anim-fade-2" style={{ overflow: 'hidden', background: 'rgba(255,255,255,.02)', borderRadius: 8, border: '1px solid var(--glass-border)' }}>
            <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
                <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                    {TABS.map(t => (
                        <button
                            key={t.key}
                            onClick={() => setTab(t.key)}
                            style={{
                                padding: '6px 12px',
                                borderRadius: 4,
                                fontSize: '.7rem',
                                fontWeight: 600,
                                cursor: 'pointer',
                                border: '1px solid transparent',
                                background: tab === t.key ? 'var(--accent-m)' : 'transparent',
                                color: tab === t.key ? 'var(--accent)' : 'var(--t4)',
                                transition: 'all .15s',
                            }}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>

                {threats.length > 1 && (
                    <select
                        value={selectedThreat}
                        onChange={e => handleThreatChange(Number(e.target.value))}
                        style={{
                            padding: '4px 8px',
                            borderRadius: 4,
                            border: '1px solid var(--brd)',
                            background: 'var(--bg-s)',
                            color: 'var(--t2)',
                            fontSize: '.68rem',
                            fontFamily: "'JetBrains Mono', monospace",
                            cursor: 'pointer',
                        }}
                    >
                        {threats.map((t, i) => (
                            <option key={i} value={i}>
                                #{t.rank} {t.debris_name} ({t.risk_level})
                            </option>
                        ))}
                    </select>
                )}
            </div>

            {/* Tab content */}
            <div style={{ padding: 16 }}>
                {loading && (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40, gap: 10 }}>
                        <div className="anim-spin" style={{ width: 20, height: 20, border: '2px solid var(--brd)', borderTopColor: 'var(--accent)', borderRadius: '50%' }} />
                        <span style={{ fontSize: '.75rem', color: 'var(--t4)' }}>{loading}</span>
                    </div>
                )}

                {error && (
                    <div style={{ padding: 20, textAlign: 'center', color: 'var(--r-high)', fontSize: '.75rem' }}>
                        {error}
                    </div>
                )}

                {!loading && !error && tab === 'risk' && currentThreat && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                        {currentThreat.temporal_risk.length > 0 && (
                            <TemporalRisk data={currentThreat.temporal_risk} />
                        )}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                            <ScatterPlot threats={threats} />
                            <UncertaintyChart threats={threats} />
                        </div>
                        <MonteCarloHist data={detailed.monte_carlo} />
                    </div>
                )}

                {!loading && !error && tab === 'bplane' && currentThreat?.bplane && (
                    <BPlaneView data={currentThreat.bplane} />
                )}
                {!loading && !error && tab === 'bplane' && currentThreat && !currentThreat.bplane && (
                    <div style={{ padding: 40, textAlign: 'center', color: 'var(--t4)', fontSize: '.75rem' }}>
                        B-plane data unavailable for this conjunction.
                    </div>
                )}

                {!loading && !error && tab === 'maneuver' && maneuverData && (
                    <ManeuverPanel data={maneuverData} />
                )}

                {!loading && !error && tab === 'interpret' && interpretData && (
                    <InterpretPanel data={interpretData} />
                )}
            </div>
        </div>
    );
}
