import { useState, useRef } from "react";
import { useNavigate } from "react-router";
import { FiUploadCloud, FiAlertCircle, FiChevronRight, FiZap, FiMap, FiBookOpen } from "react-icons/fi";

function Spinner() {
    return (
        <svg className="h-4 w-4 spin-fast" viewBox="0 0 24 24" fill="none" aria-hidden>
            <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
    );
}

const FEATURES = [
    { icon: FiZap,      label: "Instant skill gap analysis", desc: "AI reads your resume and the job description together." },
    { icon: FiMap,      label: "Personalised learning path",  desc: "A dependency graph built around your exact gaps." },
    { icon: FiBookOpen, label: "Curated resources",          desc: "Papers, courses, and repos hand-picked for your transition." },
];

export default function Home() {
    const [resumeFile, setResumeFile]       = useState(null);
    const [jobDescription, setJobDescription] = useState("");
    const [isDragging, setIsDragging]       = useState(false);
    const [isLoading, setIsLoading]         = useState(false);
    const [error, setError]                 = useState("");
    const fileInputRef = useRef(null);
    const navigate     = useNavigate();
    const backendUrl   = import.meta.env.VITE_BACKEND_URL;

    function handleFileChange(e) {
        const file = e.target.files?.[0] ?? null;
        setResumeFile(file);
        setError("");
    }

    function handleDrop(e) {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0] ?? null;
        if (file) { setResumeFile(file); setError(""); }
    }

    async function handleSubmit(e) {
        e.preventDefault();
        if (!resumeFile)      { setError("Please upload your resume."); return; }
        if (!jobDescription.trim()) { setError("Please paste the job description."); return; }

        setIsLoading(true);
        setError("");

        try {
            const formData = new FormData();
            formData.append("file", resumeFile);
            formData.append("job_description", jobDescription);
            formData.append("user_feedback", "");

            navigate("/loading");

            const response = await fetch(`${backendUrl}/analyze-skills`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const body = await response.json().catch(() => ({}));
                throw new Error(body.detail || `Server error ${response.status}`);
            }

            const result = await response.json();
            navigate("/evaluate", { state: { evaluationResult: result } });
        } catch (err) {
            navigate("/");
            setError(err.message || "Something went wrong. Please try again.");
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <main
            className="relative flex min-h-screen items-center justify-center overflow-hidden px-4 py-10 sm:px-6"
            style={{ background: "linear-gradient(145deg,#07050f 0%,#0e0a24 50%,#060d1a 100%)" }}
        >
            {/* Animated orbs */}
            <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
                <div className="animate-orb-drift absolute -top-40 -left-24 h-[480px] w-[480px] rounded-full"
                    style={{ background: "radial-gradient(circle,rgba(124,58,237,0.35) 0%,transparent 70%)", filter: "blur(72px)" }} />
                <div className="animate-orb-drift delay-300 absolute top-1/3 -right-32 h-[360px] w-[360px] rounded-full"
                    style={{ background: "radial-gradient(circle,rgba(59,130,246,0.28) 0%,transparent 70%)", filter: "blur(80px)" }} />
                <div className="animate-orb-drift delay-500 absolute -bottom-20 left-1/3 h-[300px] w-[300px] rounded-full"
                    style={{ background: "radial-gradient(circle,rgba(168,85,247,0.22) 0%,transparent 70%)", filter: "blur(64px)" }} />
                {/* Grid texture */}
                <div className="absolute inset-0 opacity-[0.04]"
                    style={{ backgroundImage: "linear-gradient(#fff 1px,transparent 1px),linear-gradient(90deg,#fff 1px,transparent 1px)", backgroundSize: "48px 48px" }} />
            </div>

            <div className="animate-fade-in relative z-10 mx-auto flex w-full max-w-5xl flex-col gap-12 lg:flex-row lg:items-center lg:gap-16">

                {/* Hero copy */}
                <div className="flex-1 text-white">
                    <span className="inline-block rounded-full border border-violet-400/30 bg-violet-400/10 px-3 py-1 text-[10px] font-bold uppercase tracking-[0.3em] text-violet-300">
                        Skill Synapse
                    </span>
                    <h1 className="mt-5 text-4xl font-bold leading-[1.12] tracking-tight sm:text-5xl lg:text-[3.25rem]">
                        Close the gap between{" "}
                        <span className="text-gradient-violet">where you are</span>{" "}
                        and where you want to be.
                    </h1>
                    <p className="mt-5 text-base leading-7 text-white/50">
                        Upload your resume, paste a job description, and get a personalised skill gap analysis with a curated learning path — in seconds.
                    </p>

                    <ul className="mt-8 grid gap-4">
                        {FEATURES.map(({ icon: Icon, label, desc }, i) => (
                            <li key={label} className={`animate-fade-in-up delay-${(i + 1) * 100} flex items-start gap-3`}>
                                <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-violet-500/20 text-violet-300">
                                    <Icon className="text-sm" />
                                </div>
                                <div>
                                    <p className="text-sm font-semibold text-white">{label}</p>
                                    <p className="text-xs text-white/40">{desc}</p>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>

                {/* Form card */}
                <div
                    className="animate-fade-in-up delay-200 w-full max-w-md shrink-0 rounded-3xl border border-white/10 p-6 shadow-2xl shadow-black/40 sm:p-8"
                    style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(20px)" }}
                >
                    <h2 className="text-lg font-semibold text-white">Analyse my skills</h2>
                    <p className="mt-1 text-xs text-white/40">Takes about 20 seconds.</p>

                    <form className="mt-6 grid gap-4" onSubmit={handleSubmit} noValidate>
                        {/* Drop zone */}
                        <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                            onDragLeave={() => setIsDragging(false)}
                            onDrop={handleDrop}
                            aria-label="Upload resume"
                            className={`w-full rounded-2xl border border-dashed px-5 py-5 text-left transition focus-violet ${
                                isDragging
                                    ? "border-violet-400 bg-violet-400/10"
                                    : resumeFile
                                    ? "border-emerald-400/50 bg-emerald-400/5"
                                    : "border-white/20 bg-white/5 hover:border-violet-400/60 hover:bg-violet-400/5"
                            }`}
                        >
                            <div className="flex items-center gap-3">
                                <FiUploadCloud className={`text-2xl shrink-0 ${resumeFile ? "text-emerald-400" : "text-white/40"}`} />
                                <div className="min-w-0">
                                    <p className={`text-sm font-medium truncate ${resumeFile ? "text-emerald-300" : "text-white/60"}`}>
                                        {resumeFile ? resumeFile.name : "Drop resume here or click to browse"}
                                    </p>
                                    <p className="text-xs text-white/30">PDF or DOCX</p>
                                </div>
                            </div>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".pdf,.doc,.docx"
                                className="sr-only"
                                onChange={handleFileChange}
                                aria-hidden
                            />
                        </button>

                        {/* Job description */}
                        <div className="grid gap-1.5">
                            <label htmlFor="jd" className="text-xs font-semibold uppercase tracking-[0.18em] text-white/50">
                                Job Description
                            </label>
                            <textarea
                                id="jd"
                                rows={6}
                                value={jobDescription}
                                onChange={(e) => { setJobDescription(e.target.value); setError(""); }}
                                placeholder="Paste the role description here…"
                                className="w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm leading-6 text-white placeholder:text-white/20 outline-none transition focus-violet focus:border-violet-400/60 focus:bg-violet-400/5"
                            />
                        </div>

                        {/* Error */}
                        {error && (
                            <div className="flex items-center gap-2 rounded-xl bg-rose-500/10 px-4 py-3 text-sm text-rose-300" role="alert">
                                <FiAlertCircle className="shrink-0" />
                                {error}
                            </div>
                        )}

                        {/* Submit */}
                        <button
                            type="submit"
                            disabled={isLoading}
                            aria-busy={isLoading}
                            className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-violet-600 to-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-900/40 transition hover:from-violet-500 hover:to-indigo-500 disabled:cursor-wait disabled:opacity-60 focus-violet"
                        >
                            {isLoading ? (
                                <><Spinner /> Analysing your résumé…</>
                            ) : (
                                <>Analyse my skills <FiChevronRight /></>
                            )}
                        </button>
                    </form>
                </div>
            </div>
        </main>
    );
}
