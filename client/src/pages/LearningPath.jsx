import { useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router";
import CytoscapeComponent from "react-cytoscapejs";
import cytoscape from "cytoscape";
import dagre from "cytoscape-dagre";
import {
    FiArrowLeft,
    FiBookOpen,
    FiCompass,
    FiExternalLink,
    FiFlag,
    FiGitBranch,
    FiTrendingUp,
} from "react-icons/fi";

cytoscape.use(dagre);

const sampleLearningPath = {
    graph: {
        nodes: [
            { id: "Python", difficulty: 3, score: 35.3, normalized_score: 1.0, time: "3 weeks", unlock_power: 10, rank: 1 },
            { id: "SQL", difficulty: 2, score: 42.86, normalized_score: 1.0, time: "2 weeks", unlock_power: 8, rank: 2 },
            { id: "Statistics", difficulty: 6, score: 8.6, normalized_score: 0.78, time: "6 weeks", unlock_power: 9, rank: 3 },
            { id: "Pandas", difficulty: 4, score: 20.0, normalized_score: 0.95, time: "2 weeks", unlock_power: 8, rank: 1 },
            { id: "Data Visualization", difficulty: 3, score: 36.0, normalized_score: 0.98, time: "1 week", unlock_power: 6, rank: 2 },
            { id: "Machine Learning", difficulty: 7, score: 2.57, normalized_score: 0.45, time: "8 weeks", unlock_power: 9, rank: 1 },
            { id: "Deep Learning", difficulty: 8, score: 1.91, normalized_score: 0.35, time: "10 weeks", unlock_power: 8, rank: 1 },
            { id: "TensorFlow", difficulty: 6, score: 3.33, normalized_score: 0.52, time: "4 weeks", unlock_power: 7, rank: 1 },
            { id: "Big Data", difficulty: 7, score: 3.43, normalized_score: 0.54, time: "6 weeks", unlock_power: 8, rank: 1 },
            { id: "Spark", difficulty: 6, score: 4.67, normalized_score: 0.62, time: "4 weeks", unlock_power: 7, rank: 1 },
        ],
        edges: [
            { from: "Python", to: "Pandas", weight: 0.95, type: "hard" },
            { from: "Python", to: "Data Visualization", weight: 0.98, type: "hard" },
            { from: "Python", to: "Machine Learning", weight: 0.45, type: "hard" },
            { from: "Python", to: "Big Data", weight: 0.54, type: "hard" },
            { from: "Python", to: "Spark", weight: 0.62, type: "hard" },
            { from: "SQL", to: "Big Data", weight: 0.54, type: "hard" },
            { from: "SQL", to: "Spark", weight: 0.62, type: "hard" },
            { from: "Statistics", to: "Machine Learning", weight: 0.45, type: "hard" },
            { from: "Statistics", to: "Deep Learning", weight: 0.35, type: "hard" },
            { from: "Pandas", to: "Machine Learning", weight: 0.45, type: "hard" },
            { from: "Machine Learning", to: "Deep Learning", weight: 0.35, type: "hard" },
            { from: "Deep Learning", to: "TensorFlow", weight: 0.52, type: "hard" },
        ],
    },
    learning_sequence: [
        "SQL",
        "Python",
        "Statistics",
        "Pandas",
        "Data Visualization",
        "Machine Learning",
        "Big Data",
        "Spark",
        "Deep Learning",
        "TensorFlow",
    ],
    tracks: {
        primary: ["SQL", "Python", "Statistics", "Machine Learning", "Deep Learning", "TensorFlow"],
        secondary: ["Pandas", "Data Visualization", "Big Data", "Spark"],
        warmup: [],
    },
    metadata: {
        total_weeks_needed: 22.5,
        time_constraint_met: true,
    },
};

const TRACK_STYLES = {
    warmup: { label: "Warmup", color: "#f59e0b" },
    primary: { label: "Primary", color: "#0ea5e9" },
    secondary: { label: "Secondary", color: "#10b981" },
    overflow: { label: "Additional", color: "#8b5cf6" },
};

function resolveTrack(id, tracks) {
    if (tracks?.warmup?.includes(id)) {
        return "warmup";
    }
    if (tracks?.primary?.includes(id)) {
        return "primary";
    }
    if (tracks?.secondary?.includes(id)) {
        return "secondary";
    }
    return "overflow";
}

function buildGraphModel(learningPath) {
    const rawNodes = learningPath?.graph?.nodes ?? [];
    const rawEdges = learningPath?.graph?.edges ?? [];
    const tracks = learningPath?.tracks ?? {};

    const nodesById = {};
    const edgesById = {};

    rawNodes.forEach((node) => {
        nodesById[node.id] = {
            id: node.id,
            label: node.id,
            track: resolveTrack(node.id, tracks),
            difficulty: Number(node.difficulty ?? 0),
            score: Number(node.score ?? 0),
            normalizedScore: Number(node.normalized_score ?? 0),
            time: node.time ?? "",
            unlockPower: Number(node.unlock_power ?? 0),
            rank: Number(node.rank ?? 0),
        };
    });

    rawEdges.forEach((edge, index) => {
        const id = `${edge.from}-${edge.to}-${index}`;
        edgesById[id] = {
            id,
            from: edge.from,
            to: edge.to,
            weight: Number(edge.weight ?? 0),
            type: edge.type ?? "hard",
        };
    });

    return { nodesById, edgesById };
}

function buildCytoscapeElements(graphModel) {
    const nodeElements = Object.values(graphModel.nodesById).map((node) => ({
        data: {
            id: node.id,
            label: node.label,
            track: node.track,
            difficulty: node.difficulty,
            score: node.score,
            normalizedScore: node.normalizedScore,
            time: node.time,
            unlockPower: node.unlockPower,
            rank: node.rank,
            color: TRACK_STYLES[node.track]?.color ?? TRACK_STYLES.overflow.color,
        },
        classes: node.track,
    }));

    const edgeElements = Object.values(graphModel.edgesById).map((edge) => ({
        data: {
            id: edge.id,
            source: edge.from,
            target: edge.to,
            label: `${Math.round(edge.weight * 100)}%`,
            weight: edge.weight,
            type: edge.type,
        },
    }));

    return [...nodeElements, ...edgeElements];
}

const cytoscapeStylesheet = [
    {
        selector: "node",
        style: {
            label: "data(label)",
            "background-color": "data(color)",
            color: "#0f172a",
            "text-wrap": "wrap",
            "text-max-width": 120,
            "text-valign": "center",
            "text-halign": "center",
            "font-size": 13,
            "font-weight": 700,
            width: 150,
            height: 54,
            shape: "round-rectangle",
            "border-width": 2,
            "border-color": "#ffffff",
            "overlay-opacity": 0,
        },
    },
    {
        selector: "edge",
        style: {
            width: 3,
            "line-color": "#475569",
            "target-arrow-color": "#475569",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "source-endpoint": "outside-to-node",
            "target-endpoint": "outside-to-node",
            label: "data(label)",
            "font-size": 10,
            color: "#475569",
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.95,
            "text-background-padding": 3,
            "text-border-color": "#cbd5e1",
            "text-border-opacity": 1,
            "text-border-width": 1,
        },
    },
    {
        selector: "node:selected",
        style: {
            "border-color": "#0284c7",
            "border-width": 4,
        },
    },
    {
        selector: "edge:selected",
        style: {
            width: 5,
            "line-color": "#0284c7",
            "target-arrow-color": "#0284c7",
        },
    },
];

export default function LearningPath() {
    const { state } = useLocation();
    const navigate = useNavigate();
    const evaluationResult = state?.evaluationResult;
    const selfRatings = state?.selfRatings ?? {};
    const learningPath = state?.learningPathResult ?? sampleLearningPath;
    const cyRef = useRef(null);

    const graphModel = useMemo(() => buildGraphModel(learningPath), [learningPath]);
    const elements = useMemo(() => buildCytoscapeElements(graphModel), [graphModel]);
    const [selectedItem, setSelectedItem] = useState(() => {
        const firstNode = Object.values(graphModel.nodesById)[0];
        return firstNode ? { type: "node", data: firstNode } : null;
    });

    const hasSelfRatings = Object.keys(selfRatings).length > 0;
    const totalNodes = Object.keys(graphModel.nodesById).length;
    const totalEdges = Object.keys(graphModel.edgesById).length;
    const totalWeeks = learningPath?.metadata?.total_weeks_needed ?? 0;
    const timeConstraintMet = Boolean(learningPath?.metadata?.time_constraint_met);

    function handleGraphItemClick(item) {
        setSelectedItem(item);
        const search =
            item.type === "node"
                ? `?id=${encodeURIComponent(item.data.id)}`
                : `?from=${encodeURIComponent(item.data.from)}&to=${encodeURIComponent(item.data.to)}`;

        navigate(`/learning-path/resources${search}`, {
            state: {
                evaluationResult,
                learningPathResult: learningPath,
                selfRatings,
            },
        });
    }

    return (
        <main className="min-h-screen bg-[radial-gradient(circle_at_top,#f8fafc,#dbeafe_38%,#cbd5e1_72%,#0f172a_150%)] px-4 py-4 text-slate-900 sm:px-6 sm:py-6">
            <div className="mx-auto flex max-w-7xl flex-col gap-5">
                <header className="rounded-4xl border border-white/60 bg-white/80 p-5 shadow-xl shadow-slate-900/5 backdrop-blur sm:p-6">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                        <div className="max-w-4xl">
                            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-700">
                                Skill Synapse
                            </p>
                            <h1 className="mt-2 font-serif text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">
                                Interactive Learning Path
                            </h1>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                Rebuilt with a dedicated graph library so the pathway
                                behaves like an actual graph, with proper edge routing,
                                node spacing, and click handling.
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

                <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                    <article className="rounded-4xl border border-slate-200/70 bg-slate-950 p-4 text-white shadow-xl shadow-slate-900/10">
                        <div className="rounded-2xl bg-sky-400/15 p-3 text-sky-300">
                            <FiCompass className="text-lg" />
                        </div>
                        <p className="mt-4 text-[11px] uppercase tracking-[0.2em] text-slate-400">
                            Skills in graph
                        </p>
                        <p className="mt-1 text-2xl font-semibold">{totalNodes}</p>
                    </article>

                    <article className="rounded-4xl border border-slate-200/70 bg-slate-950 p-4 text-white shadow-xl shadow-slate-900/10">
                        <div className="rounded-2xl bg-emerald-400/15 p-3 text-emerald-300">
                            <FiGitBranch className="text-lg" />
                        </div>
                        <p className="mt-4 text-[11px] uppercase tracking-[0.2em] text-slate-400">
                            Dependencies
                        </p>
                        <p className="mt-1 text-2xl font-semibold">{totalEdges}</p>
                    </article>

                    <article className="rounded-4xl border border-slate-200/70 bg-slate-950 p-4 text-white shadow-xl shadow-slate-900/10">
                        <div className="rounded-2xl bg-amber-400/15 p-3 text-amber-300">
                            <FiFlag className="text-lg" />
                        </div>
                        <p className="mt-4 text-[11px] uppercase tracking-[0.2em] text-slate-400">
                            Total weeks
                        </p>
                        <p className="mt-1 text-2xl font-semibold">{totalWeeks}</p>
                    </article>

                    <article className="rounded-4xl border border-slate-200/70 bg-slate-950 p-4 text-white shadow-xl shadow-slate-900/10">
                        <div className="rounded-2xl bg-fuchsia-400/15 p-3 text-fuchsia-300">
                            <FiTrendingUp className="text-lg" />
                        </div>
                        <p className="mt-4 text-[11px] uppercase tracking-[0.2em] text-slate-400">
                            Flow state
                        </p>
                        <p className="mt-1 text-lg font-semibold">
                            {timeConstraintMet ? "On track" : "Needs review"}
                        </p>
                        <p className="mt-1 text-xs text-slate-400">
                            {hasSelfRatings ? "Includes self-ratings" : "Graph only route entry"}
                        </p>
                    </article>
                </section>

                <section className="rounded-4xl border border-slate-200/70 bg-white/90 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                    <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div>
                            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                                Learning graph
                            </p>
                            <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
                                Skill dependency pathway
                            </h2>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {Object.entries(TRACK_STYLES).map(([key, value]) => (
                                <span
                                    key={key}
                                    className="rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em]"
                                    style={{
                                        backgroundColor: `${value.color}18`,
                                        color: value.color,
                                    }}
                                >
                                    {value.label}
                                </span>
                            ))}
                        </div>
                    </div>

                    <div className="mt-5 h-[70vh] min-h-160 rounded-4xl border border-slate-200 bg-[linear-gradient(180deg,#f8fafc,#eef6ff)]">
                        <CytoscapeComponent
                            elements={elements}
                            stylesheet={cytoscapeStylesheet}
                            style={{ width: "100%", height: "100%" }}
                            layout={{
                                name: "dagre",
                                rankDir: "TB",
                                padding: 40,
                                nodeSep: 50,
                                rankSep: 100,
                                edgeSep: 18,
                                animate: false,
                            }}
                            cy={(cy) => {
                                cyRef.current = cy;
                                cy.off("tap", "node");
                                cy.off("tap", "edge");
                                cy.on("tap", "node", (event) => {
                                    const data = event.target.data();
                                    handleGraphItemClick({ type: "node", data });
                                });
                                cy.on("tap", "edge", (event) => {
                                    const data = event.target.data();
                                    handleGraphItemClick({
                                        type: "edge",
                                        data: {
                                            id: data.id,
                                            from: data.source,
                                            to: data.target,
                                            weight: data.weight,
                                            type: data.type,
                                        },
                                    });
                                });
                            }}
                            userZoomingEnabled={false}
                            userPanningEnabled={false}
                            boxSelectionEnabled={false}
                            autoungrabify={true}
                            autounselectify={false}
                        />
                    </div>
                </section>

                <section className="grid gap-4 xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
                    <section className="rounded-4xl border border-slate-200/70 bg-white/90 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                            Selection
                        </p>
                        {selectedItem?.type === "node" ? (
                            <>
                                <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
                                    {selectedItem.data.label}
                                </h2>
                                <p className="mt-3 text-sm leading-6 text-slate-600">
                                    Clicking a skill opens `/learning-path/resources?id=...`.
                                </p>
                                <div className="mt-5 grid gap-3 text-sm text-slate-600">
                                    <div className="rounded-3xl bg-slate-50 p-4">
                                        Track:{" "}
                                        <span className="font-semibold text-slate-900">
                                            {TRACK_STYLES[selectedItem.data.track]?.label ?? "Additional"}
                                        </span>
                                    </div>
                                    <div className="rounded-3xl bg-slate-50 p-4">
                                        Time:{" "}
                                        <span className="font-semibold text-slate-900">
                                            {selectedItem.data.time}
                                        </span>
                                    </div>
                                    <div className="rounded-3xl bg-slate-50 p-4">
                                        Unlock power:{" "}
                                        <span className="font-semibold text-slate-900">
                                            {selectedItem.data.unlockPower}
                                        </span>
                                    </div>
                                    <div className="rounded-3xl bg-slate-50 p-4">
                                        Difficulty:{" "}
                                        <span className="font-semibold text-slate-900">
                                            {selectedItem.data.difficulty}/10
                                        </span>
                                    </div>
                                </div>
                            </>
                        ) : selectedItem?.type === "edge" ? (
                            <>
                                <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
                                    {selectedItem.data.from} to {selectedItem.data.to}
                                </h2>
                                <p className="mt-3 text-sm leading-6 text-slate-600">
                                    Clicking a dependency opens `/learning-path/resources?from=...&to=...`.
                                </p>
                                <div className="mt-5 grid gap-3 text-sm text-slate-600">
                                    <div className="rounded-3xl bg-slate-50 p-4">
                                        Type:{" "}
                                        <span className="font-semibold text-slate-900">
                                            {selectedItem.data.type}
                                        </span>
                                    </div>
                                    <div className="rounded-3xl bg-slate-50 p-4">
                                        Strength:{" "}
                                        <span className="font-semibold text-slate-900">
                                            {Math.round(selectedItem.data.weight * 100)}%
                                        </span>
                                    </div>
                                </div>
                            </>
                        ) : (
                            <p className="mt-3 text-sm leading-6 text-slate-600">
                                Click any node or edge in the graph to inspect it here.
                            </p>
                        )}

                        <div className="mt-6 rounded-3xl border border-slate-200 bg-slate-50 p-4">
                            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700">
                                Resource routing
                            </p>
                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                Nodes navigate with `id=...`. Edges navigate with `from=...&to=...`.
                                The resource page uses that URL to fetch backend data.
                            </p>
                            <p className="mt-3 inline-flex items-center gap-2 rounded-full bg-slate-950 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-white">
                                <FiBookOpen className="text-sm" />
                                Sharable resource URL
                            </p>
                        </div>
                    </section>

                    <section className="rounded-4xl border border-slate-200/70 bg-white/90 p-5 shadow-xl shadow-slate-900/5 backdrop-blur">
                        <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">
                            Reading guide
                        </p>
                        <div className="mt-4 grid gap-3 text-sm leading-6 text-slate-600">
                            <div className="rounded-3xl bg-slate-50 p-4">
                                The graph is laid out as a directed dependency map from top to bottom.
                            </div>
                            <div className="rounded-3xl bg-slate-50 p-4">
                                Node color indicates track membership.
                            </div>
                            <div className="rounded-3xl bg-slate-50 p-4">
                                Edge labels show dependency strength.
                            </div>
                            <div className="rounded-3xl bg-slate-50 p-4">
                                Nodes and edges now open dedicated resource routes on click.
                            </div>
                        </div>
                        <div className="mt-5 flex justify-end">
                            <button
                                type="button"
                                disabled
                                className="inline-flex cursor-not-allowed items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-500"
                            >
                                <FiExternalLink className="text-base" />
                                Resource launcher next
                            </button>
                        </div>
                    </section>
                </section>
            </div>
        </main>
    );
}
