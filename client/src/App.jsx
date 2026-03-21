import { createBrowserRouter } from "react-router";
import { RouterProvider } from "react-router/dom";
import Home from "./pages/Home.jsx";
import Evaluate from "./pages/Evaluate.jsx";
import SkillRatings from "./pages/SkillRatings.jsx";
import LearningPath from "./pages/LearningPath.jsx";
import NotFound from "./pages/NotFound.jsx";

const router = createBrowserRouter([
    {
        path: "/",
        element: <Home />,
    },
    {
        path: "/evaluate",
        element: <Evaluate />,
    },
    {
        path: "/skill-rating",
        element: <SkillRatings />,
    },
    {
        path: "/learning-path",
        element: <LearningPath />,
    },
    {
        path: "*",
        element: <NotFound />,
    },
]);

export function App() {
    return <RouterProvider router={router} />;
}
