export default function Loading() {
    return (
        <main
            className="relative flex min-h-screen items-center justify-center overflow-hidden px-6"
            style={{ background: "linear-gradient(145deg,#07050f 0%,#0e0a24 50%,#060d1a 100%)" }}
        >
            {/* Orbs */}
            <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
                <div
                    className="absolute -top-32 left-1/4 h-80 w-80 rounded-full"
                    style={{ background: "radial-gradient(circle,rgba(124,58,237,0.4) 0%,transparent 70%)", filter: "blur(60px)" }}
                />
                <div
                    className="absolute bottom-0 right-1/4 h-64 w-64 rounded-full"
                    style={{ background: "radial-gradient(circle,rgba(59,130,246,0.3) 0%,transparent 70%)", filter: "blur(70px)" }}
                />
            </div>

            <section
                className="animate-fade-in relative flex w-full max-w-xs flex-col items-center rounded-3xl border border-white/10 px-8 py-10 text-center text-white"
                style={{ background: "rgba(255,255,255,0.05)", backdropFilter: "blur(20px)" }}
                aria-live="polite"
                aria-label="Loading"
            >
                {/* Spinner */}
                <div className="relative flex h-20 w-20 items-center justify-center" aria-hidden>
                    <span className="absolute inline-block h-20 w-20 animate-ping rounded-full bg-violet-400/15" />
                    <span className="absolute inline-block h-14 w-14 rounded-full border border-violet-400/20" />
                    <svg className="h-10 w-10 spin-fast" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-15" cx="12" cy="12" r="10" stroke="white" strokeWidth="3" />
                        <path className="opacity-80" fill="white" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                </div>

                <p className="mt-7 text-[10px] font-bold uppercase tracking-[0.35em] text-violet-300">
                    Skill Synapse
                </p>
                <h1 className="mt-2 text-xl font-semibold tracking-tight">Preparing your results</h1>
                <p className="mt-2.5 text-sm leading-6 text-white/40">
                    Analysing skills and building your personalised path.
                </p>

                {/* Progress dots */}
                <div className="mt-7 flex gap-2" aria-hidden>
                    {[0, 1, 2].map((i) => (
                        <span
                            key={i}
                            className="h-1.5 w-1.5 rounded-full bg-violet-400/60"
                            style={{ animation: `pulse 1.4s ease-in-out ${i * 0.24}s infinite` }}
                        />
                    ))}
                </div>
            </section>
        </main>
    );
}
