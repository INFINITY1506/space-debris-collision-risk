import { useState, useEffect } from 'react';
import './index.css';
import { SearchBar } from './components/SearchBar';
import { ResultsTable } from './components/ResultsTable';
import { SatelliteCard } from './components/SatelliteCard';
import { GlobeView } from './components/GlobeView';
import { LoadingSpinner, ErrorMessage, StatCard, RiskSummaryBar } from './components/LoadingSpinner';
import { predictCollision, predictDetailed, getHealth, type PredictResponse, type DetailedPredictResponse, type HealthResponse } from './api/client';
import { AnalyticsDashboard } from './components/AnalyticsDashboard';
import { TopBar } from './components/layout/TopBar';
import { BottomBar } from './components/layout/BottomBar';
import { Sidebar } from './components/layout/Sidebar';
import { RightPanel } from './components/layout/RightPanel';

export default function App() {
    const [result, setResult] = useState<PredictResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [lastQuery, setLastQuery] = useState('');
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [detailed, setDetailed] = useState<DetailedPredictResponse | null>(null);
    const [detailedLoading, setDetailedLoading] = useState(false);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [rightPanelOpen, setRightPanelOpen] = useState(false);

    useEffect(() => { getHealth().then(setHealth).catch(() => setHealth(null)); }, []);

    const handleSearch = async (name: string) => {
        setLoading(true); setError(null); setResult(null); setDetailed(null); setLastQuery(name);
        try {
            const noradId = /^\d+$/.test(name.trim()) ? parseInt(name.trim()) : undefined;
            const data = await predictCollision(noradId ? undefined : name, noradId, 10);
            setResult(data);
            setRightPanelOpen(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unexpected error occurred');
        } finally { setLoading(false); }
    };

    const handleDetailedAnalysis = async () => {
        if (!result) return;
        setDetailedLoading(true);
        try {
            const noradId = result.satellite.norad_id;
            const data = await predictDetailed(undefined, noradId, 10, 50);
            setDetailed(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Detailed analysis failed');
        } finally { setDetailedLoading(false); }
    };

    const online = !!health?.model_loaded;
    const topPct = result?.threats?.[0]?.collision_probability ?? 0;
    const highN = result?.threats?.filter(t => t.risk_level === 'HIGH').length ?? 0;
    const medN  = result?.threats?.filter(t => t.risk_level === 'MEDIUM').length ?? 0;

    return (
        <>
            {/* Full-screen globe background */}
            <GlobeView onSelectSatellite={handleSearch} selectedSatName={result?.satellite?.name} />

            {/* Top bar */}
            <TopBar
                online={online}
                health={health}
                sidebarCollapsed={sidebarCollapsed}
                onToggleSidebar={() => setSidebarCollapsed(c => !c)}
            />

            {/* Left sidebar — search & satellite info */}
            <Sidebar collapsed={sidebarCollapsed}>
                <SearchBar onSearch={handleSearch} loading={loading} />

                {result && !loading && (
                    <>
                        <SatelliteCard satellite={result.satellite} time={result.total_time_s} pairs={result.n_candidates_analyzed} />

                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            <StatCard label="Top Threat" value={`${topPct.toFixed(4)}%`}
                                color={topPct > .01 ? 'var(--r-high)' : topPct > .001 ? 'var(--r-med)' : 'var(--r-low)'}
                                sub={highN > 0 ? `${highN} HIGH` : medN > 0 ? `${medN} MED` : 'All LOW'} />
                            <StatCard label="Threats" value={result.threats.length} color="var(--accent)"
                                sub={`of ${result.n_candidates_analyzed.toLocaleString()}`} />
                            <StatCard label="Propagation" value={`${result.propagation_time_s?.toFixed(2) ?? '\u2014'}s`}
                                color="var(--t2)" sub="SGP4 orbits" />
                            <StatCard label="Inference" value={`${result.inference_time_s?.toFixed(2) ?? '\u2014'}s`}
                                color="var(--t2)" sub="Transformer" />
                            {result.threats.length > 0 && <RiskSummaryBar threats={result.threats} />}
                        </div>

                        {!rightPanelOpen && result.threats.length > 0 && (
                            <button
                                onClick={() => setRightPanelOpen(true)}
                                style={{
                                    width: '100%', padding: '8px 12px', borderRadius: 6,
                                    border: '1px solid var(--accent)', background: 'var(--accent-m)',
                                    color: 'var(--accent)', fontSize: '.72rem', fontWeight: 600,
                                    cursor: 'pointer', letterSpacing: '.03em',
                                }}
                            >
                                View Results &rarr;
                            </button>
                        )}
                    </>
                )}
            </Sidebar>

            {/* Right panel — results & analytics */}
            <RightPanel open={rightPanelOpen && !!result && !loading} onClose={() => setRightPanelOpen(false)}>
                {result && (
                    <>
                        {result.threats.length > 0 ? (
                            <ResultsTable threats={result.threats} satellite={result.satellite}
                                meta={{ pairs: result.n_candidates_analyzed, totalTime: result.total_time_s, inferTime: result.inference_time_s }} />
                        ) : (
                            <div style={{ padding: 20, textAlign: 'center', color: 'var(--t3)', fontSize: '.8rem' }}>
                                No threats detected for {result.satellite.name}
                            </div>
                        )}

                        {result.threats.length > 0 && !detailed && (
                            <button
                                onClick={handleDetailedAnalysis}
                                disabled={detailedLoading}
                                style={{
                                    width: '100%', padding: '10px 16px', borderRadius: 6,
                                    border: '1px solid var(--accent)', background: 'var(--accent-m)',
                                    color: 'var(--accent)', fontSize: '.75rem', fontWeight: 600,
                                    cursor: detailedLoading ? 'wait' : 'pointer', letterSpacing: '.03em',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                                }}
                            >
                                {detailedLoading ? (
                                    <>
                                        <span className="anim-spin" style={{ display: 'inline-block', width: 14, height: 14, border: '2px solid var(--brd)', borderTopColor: 'var(--accent)', borderRadius: '50%' }} />
                                        Running Detailed Analysis...
                                    </>
                                ) : (
                                    <>
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /><path d="M10 7v6m-3-3h6" /></svg>
                                        Advanced Analytics
                                    </>
                                )}
                            </button>
                        )}

                        {detailed && <AnalyticsDashboard detailed={detailed} />}
                    </>
                )}
            </RightPanel>

            {/* Bottom bar */}
            <BottomBar />

            {/* Loading overlay */}
            {loading && (
                <div className="glass-modal">
                    <div className="glass-modal-content">
                        <LoadingSpinner message="Analyzing collision risk..." subMessage={`Propagating ${lastQuery} against catalog debris`} />
                    </div>
                </div>
            )}

            {/* Error overlay */}
            {error && (
                <div className="glass-modal" onClick={() => setError(null)}>
                    <div className="glass-modal-content" onClick={e => e.stopPropagation()}>
                        <ErrorMessage message={error} onRetry={() => { setError(null); lastQuery && handleSearch(lastQuery); }} />
                        <button onClick={() => setError(null)}
                            style={{ marginTop: 8, padding: '5px 14px', fontSize: '.72rem', background: 'none', border: '1px solid var(--brd)', borderRadius: 4, color: 'var(--t3)', cursor: 'pointer' }}>
                            Dismiss
                        </button>
                    </div>
                </div>
            )}
        </>
    );
}
