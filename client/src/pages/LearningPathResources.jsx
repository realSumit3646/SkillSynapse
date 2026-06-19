import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useSearchParams } from "react-router";
import {
    FiArrowLeft,
    FiBookOpen,
    FiCode,
    FiExternalLink,
    FiFileText,
    FiGlobe,
    FiPlayCircle,
} from "react-icons/fi";

const RESOURCE_SECTIONS = [
    { key: "research_papers", label: "Research papers", icon: FiFileText },
    { key: "books", label: "Books", icon: FiBookOpen },
    { key: "github", label: "GitHub", icon: FiCode },
    { key: "youtube", label: "YouTube", icon: FiPlayCircle },
    { key: "websites", label: "Websites", icon: FiGlobe },
    { key: "documentation", label: "Documentation", icon: FiFileText },
];


export default function LearningPathResources() {
    const { state } = useLocation();
    const [searchParams] = useSearchParams();
    const backendUrl = import.meta.env.VITE_BACKEND_URL;
    const id = searchParams.get("id");
    const from = searchParams.get("from");
    const to = searchParams.get("to");
    const isNodeRoute = Boolean(id);
    const requestPayload = useMemo(() => {
        if (id) {
            return { from: id, to: id };
        }
        if (from && to) {
            return { from, to };
        }
        return null;
    }, [from, id, to]);
    const [requestState, setRequestState] = useState({
        loading: true,
        error: "",
        data: null,
    });

    useEffect(() => {
        let active = true;

        async function loadResources() {
            if (!backendUrl) {
                setRequestState({
                    loading: false,
                    error: "VITE_BACKEND_URL is not configured.",
                    data: null,
                });
                return;
            }

            if (!requestPayload) {
                setRequestState({
                    loading: false,
                    error: "Missing resource query. Use id=... or from=...&to=....",
                    data: null,
                });
                return;
            }

            setRequestState({ loading: true, error: "", data: null });

            try {
                const response = await fetch(`${backendUrl}/get-resources`, {
                    method: "POST",
                    body: JSON.stringify(requestPayload),
                    headers: {
                        "Content-Type": "application/json",
                    },
                });

                if (!response.ok) {
                    throw new Error("Unable to load learning resources.");
                }

                const result = await response.json();
                if (active) {
                    setRequestState({ loading: false, error: "", data: result });
                }
            } catch (error) {
                if (active) {
                    setRequestState({
                        loading: false,
                        error: error instanceof Error ? error.message : "Request failed.",
                        data: null,
                    });
                }
            }
        }

        loadResources();

        return () => {
            active = false;
        };
    }, [backendUrl, requestPayload]);

    const result = requestState.data?.result;
    const title = isNodeRoute
        ? `Resources for ${id ?? "skill"}`
        : `Resources from ${from ?? "skill"} to ${to ?? "skill"}`;

    return (
        <main className="min-h-screen bg-[radial-gradient(ellipse_at_top_left,#ede9fe_0%,#f8f7ff_30%,#f1f5f9_65%,#e2e8f0_100%)] px-4 py-4 text-slate-900 sm:px-6 sm:py-6">
            <div className="mx-auto flex max-w-7xl flex-col gap-5">
                <header className="rounded-4xl border border-white/60 bg-white/85 p-5 shadow-xl shadow-slate-900/5 backdrop-blur sm:p-6">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                        <div className="max-w-4xl">
                            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-700">
                                Skill Synapse
                            </p>
                            <h1 className="mt-2 font-serif text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">
                                {title}
                            </h1>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                This page is driven by the URL query, so each node and edge has a
                                direct resource route you can revisit or share.
                            </p>
                        </div>

                        <Link
                            to="/learning-path"
                            state={state}
                            className="inline-flex items-center gap-2 self-start rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-sky-300 hover:text-sky-700"
                        >
                            <FiArrowLeft className="text-base" />
                            Back to graph
                        </Link>
                    </div>
                </header>

                <section className="rounded-4xl border border-slate-200/70 bg-white/90 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                    <div className="flex flex-wrap items-center gap-3">
                        <span className="rounded-full bg-slate-950 px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-white">
                            {isNodeRoute ? `Topic: ${id}` : `Transition: ${from} to ${to}`}
                        </span>
                    </div>
                </section>

                {requestState.loading ? (
                    <section className="flex flex-col items-center gap-4 rounded-4xl border border-slate-200/70 bg-white/90 p-12 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <svg className="h-8 w-8 spin-fast text-violet-500" viewBox="0 0 24 24" fill="none" aria-hidden>
                            <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                            <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        <p className="text-sm font-medium text-slate-500">Fetching curated resources — this may take a moment…</p>
                    </section>
                ) : requestState.error ? (
                    <section className="rounded-4xl border border-rose-200 bg-rose-50 p-8 text-sm text-rose-700 shadow-xl shadow-rose-200/40">
                        {requestState.error}
                    </section>
                ) : (
                    <section className="grid gap-4 lg:grid-cols-2">
                        {RESOURCE_SECTIONS.map(({ key, label, icon }) => {
                            const items = result?.resources?.[key] ?? [];
                            const IconComponent = icon;

                            return (
                                <article
                                    key={key}
                                    className="rounded-4xl border border-slate-200/70 bg-white/90 p-5 shadow-xl shadow-slate-900/5 backdrop-blur"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className="rounded-2xl bg-sky-50 p-3 text-sky-700">
                                            <IconComponent className="text-lg" />
                                        </div>
                                        <div>
                                            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700">
                                                {label}
                                            </p>
                                            <h2 className="mt-1 text-xl font-semibold text-slate-950">
                                                {items.length} suggestions
                                            </h2>
                                        </div>
                                    </div>

                                    <div className="mt-5 grid gap-3">
                                        {items.map((item) => (
                                            <a
                                                key={`${key}-${item.url}`}
                                                href={item.url}
                                                target="_blank"
                                                rel="noreferrer"
                                                className="group rounded-[1.75rem] border border-slate-200 bg-slate-50 p-4 transition hover:border-sky-300 hover:bg-sky-50"
                                            >
                                                <div className="flex items-start justify-between gap-3">
                                                    <div className="min-w-0 flex-1">
                                                        <h3 className="text-base font-semibold text-slate-950">
                                                            {item.title}
                                                        </h3>
                                                        <p className="mt-1 text-sm text-slate-500">
                                                            {item.source}
                                                        </p>
                                                        <div className="mt-4 flex flex-wrap gap-2 text-xs font-semibold uppercase tracking-[0.12em]">
                                                            <span className="rounded-full bg-slate-950 px-3 py-1 text-white">
                                                                {item.level}
                                                            </span>
                                                            <span className="rounded-full bg-violet-50 px-3 py-1 text-violet-700">
                                                                {item.source}
                                                            </span>
                                                        </div>
                                                    </div>
                                                    <div className="flex shrink-0 items-start gap-3">
                                                        {item.image_url ? (
                                                            <img
                                                                src={item.image_url}
                                                                alt={item.title}
                                                                className="h-16 w-16 rounded-2xl object-cover"
                                                            />
                                                        ) : null}
                                                        <FiExternalLink className="mt-1 shrink-0 text-slate-400 transition group-hover:text-sky-700" />
                                                    </div>
                                                </div>
                                            </a>
                                        ))}
                                    </div>
                                </article>
                            );
                        })}
                    </section>
                )}
            </div>
        </main>
    );
}
