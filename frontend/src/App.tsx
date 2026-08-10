import { lazy, Suspense, useState, useEffect } from 'react';
import './index.css';
import { SearchBar } from './components/SearchBar';
import { ResultsTable } from './components/ResultsTable';
import { SatelliteCard } from './components/SatelliteCard';
import { LoadingSpinner, ErrorMessage, StatCard, RiskSummaryBar } from './components/LoadingSpinner';
import { predictCollision, predictDetailed, getHealth, type PredictResponse, type DetailedPredictResponse, type HealthResponse } from './api/client';
import { TopBar } from './components/layout/TopBar';
import { BottomBar } from './components/layout/BottomBar';
import { Sidebar } from './components/layout/Sidebar';
import { RightPanel } from './components/layout/RightPanel';
import { AboutPanel } from './components/AboutPanel';

const GlobeView = lazy(() => import('./components/GlobeView').then(module => ({ default: module.GlobeView })));
const AnalyticsDashboard = lazy(() => import('./components/AnalyticsDashboard').then(module => ({ default: module.AnalyticsDashboard })));

export default function App() {
    const [result, setResult] = useState<PredictResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [loadingPhase, setLoadingPhase] = useState<'idle' | 'warming' | 'analyzing'>('idle');
    const [error, setError] = useState<string | null>(null);
    const [lastQuery, setLastQuery] = useState('');
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [detailed, setDetailed] = useState<DetailedPredictResponse | null>(null);
    const [detailedLoading, setDetailedLoading] = useState(false);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(
        () => typeof window !== 'undefined' && window.innerWidth <= 640,
    );
    const [rightPanelOpen, setRightPanelOpen] = useState(false);
    const [aboutOpen, setAboutOpen] = useState(false);

    useEffect(() => {
        let cancelled = false;
        const checkHealth = () => getHealth()
            .then(data => { if (!cancelled) setHealth(data); })
            .catch(() => { if (!cancelled) setHealth(null); });

        checkHealth();
        const timer = window.setInterval(checkHealth, 5000);
        return () => {
            cancelled = true;
            window.clearInterval(timer);
        };
    }, []);

    const handleSearch = async (name: string) => {
        setLoading(true);
        setLoadingPhase(health?.model_loaded ? 'analyzing' : 'warming');
        setError(null); setResult(null); setDetailed(null); setLastQuery(name);
        try {
            const noradId = /^\d+$/.test(name.trim()) ? parseInt(name.trim()) : undefined;
            const data = await predictCollision(noradId ? undefined : name, noradId, 10, {
                onWarmup: () => setLoadingPhase('warming'),
            });
            setResult(data);
            setRightPanelOpen(true);
            if (window.innerWidth < 900) setSidebarCollapsed(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unexpected error occurred');
        } finally {
            setLoading(false);
            setLoadingPhase('idle');
        }
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

    const online = health?.screening_available ?? !!health?.model_loaded;
    const topPct = result?.threats?.[0]?.collision_probability ?? 0;
    const highN = result?.threats?.filter(t => t.risk_level === 'HIGH').length ?? 0;
    const medN  = result?.threats?.filter(t => t.risk_level === 'MEDIUM').length ?? 0;

    return (
        <>
            {/* Full-screen globe background */}
            <Suspense fallback={<div className="globe-loading">Loading interactive globe…</div>}>
                <GlobeView onSelectSatellite={handleSearch} selectedSatName={result?.satellite?.name} />
            </Suspense>

            {!result && !loading && !error && (
                <section className={`mission-intro ${sidebarCollapsed ? 'mission-intro--wide' : ''}`}>
                    <div className="mission-intro__index">DS / 01</div>
                    <div className="mission-intro__eyebrow">
                        <span>Near-Earth object screening</span>
                        <span>Seven-day horizon</span>
                    </div>
                    <h1>See the crowded<br />orbit.</h1>
                    <p>
                        Select an active spacecraft to screen public debris trajectories,
                        rank close approaches, and inspect the geometry behind each result.
                    </p>
                    <div className="mission-intro__facts">
                        <span><b>SGP4</b> propagation</span>
                        <span><b>{health?.catalog_size ? health.catalog_size.toLocaleString() : '—'}</b> tracked objects</span>
                        <span><b>Public</b> TLE data</span>
                    </div>
                </section>
            )}

            {/* Top bar */}
            <TopBar
                online={online}
                health={health}
                sidebarCollapsed={sidebarCollapsed}
                onToggleSidebar={() => setSidebarCollapsed(c => !c)}
                onOpenAbout={() => setAboutOpen(true)}
            />

            {/* Left sidebar — search & satellite info */}
            <Sidebar collapsed={sidebarCollapsed}>
                <div className="console-heading">
                    <span className="console-heading__step">01 / SCREEN</span>
                    <h2>Conjunction query</h2>
                    <p>Choose an active spacecraft or enter a NORAD catalog number.</p>
                </div>
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
                                    width: '100%', padding: '8px 12px', borderRadius: 2,
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
                                    width: '100%', padding: '10px 16px', borderRadius: 2,
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

                        {detailed && (
                            <Suspense fallback={<LoadingSpinner message="Loading analytics…" />}>
                                <AnalyticsDashboard detailed={detailed} />
                            </Suspense>
                        )}
                    </>
                )}
            </RightPanel>

            {/* Bottom bar */}
            <BottomBar />

            <AboutPanel open={aboutOpen} onClose={() => setAboutOpen(false)} />

            {/* Loading overlay */}
            {loading && (
                <div className="glass-modal">
                    <div className="glass-modal-content">
                        <LoadingSpinner
                            message={loadingPhase === 'warming' ? 'Waking up Debris Sentinel...' : 'Analyzing collision risk...'}
                            subMessage={loadingPhase === 'warming'
                                ? 'Cloud Run is refreshing orbital data and loading the model. This can take up to two minutes after inactivity.'
                                : `Propagating ${lastQuery} against catalog debris`}
                        />
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
