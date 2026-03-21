import { Link, useLocation } from "react-router";
import { FiArrowLeft, FiCompass, FiFlag, FiTrendingUp } from "react-icons/fi";

export default function LearningPath() {
    const { state } = useLocation();
    const evaluationResult = state?.evaluationResult;
    const selfRatings = state?.selfRatings ?? {};
    const hasSelfRatings = Object.keys(selfRatings).length > 0;

    return (
        <main className="min-h-screen bg-[radial-gradient(circle_at_top,#f8fafc,#dbeafe_38%,#cbd5e1_72%,#0f172a_150%)] px-4 py-4 text-slate-900 sm:px-6 sm:py-6">
            <div className="mx-auto flex max-w-6xl flex-col gap-5">
                <header className="rounded-4xl border border-white/60 bg-white/80 p-5 shadow-xl shadow-slate-900/5 backdrop-blur sm:p-6">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                        <div className="max-w-3xl">
                            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-700">
                                Skill Synapse
                            </p>
                            <h1 className="mt-2 font-serif text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">
                                Learning Path
                            </h1>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                This route is now connected in the flow. It is ready
                                to receive the evaluation result and any self-ratings
                                before the learning path content is finalized.
                            </p>
                        </div>

                        <Link
                            to="/evaluate"
                            state={{ evaluationResult }}
                            className="inline-flex items-center gap-2 self-start rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-sky-300 hover:text-sky-700"
                        >
                            <FiArrowLeft className="text-base" />
                            Back to dashboard
                        </Link>
                    </div>
                </header>

                <section className="grid gap-4 md:grid-cols-3">
                    <article className="rounded-4xl border border-slate-200/70 bg-white/85 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <div className="rounded-2xl bg-sky-50 p-3 text-sky-700">
                            <FiCompass className="text-lg" />
                        </div>
                        <h2 className="mt-4 text-xl font-semibold text-slate-950">
                            Flow connected
                        </h2>
                        <p className="mt-2 text-sm leading-6 text-slate-600">
                            Users can now land here either directly from the evaluation
                            page or after completing the self-rating step.
                        </p>
                    </article>

                    <article className="rounded-4xl border border-slate-200/70 bg-white/85 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <div className="rounded-2xl bg-emerald-50 p-3 text-emerald-700">
                            <FiTrendingUp className="text-lg" />
                        </div>
                        <h2 className="mt-4 text-xl font-semibold text-slate-950">
                            Self-ratings
                        </h2>
                        <p className="mt-2 text-sm leading-6 text-slate-600">
                            {hasSelfRatings
                                ? "Self-ratings were passed into this route and are ready for downstream use."
                                : "No self-ratings were needed for this route entry, which is expected when there are no flagged gaps."}
                        </p>
                    </article>

                    <article className="rounded-4xl border border-slate-200/70 bg-white/85 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <div className="rounded-2xl bg-amber-50 p-3 text-amber-700">
                            <FiFlag className="text-lg" />
                        </div>
                        <h2 className="mt-4 text-xl font-semibold text-slate-950">
                            Next build step
                        </h2>
                        <p className="mt-2 text-sm leading-6 text-slate-600">
                            This is the place to generate and display the actual learning
                            path content when you are ready to connect that logic.
                        </p>
                    </article>
                </section>
            </div>
        </main>
    );
}
