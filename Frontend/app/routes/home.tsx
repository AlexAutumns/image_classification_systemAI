import React, { useState } from "react";

type TopItem = { label: string; confidence: number };
type Bucket = {
    predicted: string;
    confidence: number;
    top3: TopItem[];
    top3Restricted?: TopItem[];
};
interface ClassificationResult {
    master: Bucket;
    sub: Bucket;
    article: Bucket;
    hierarchyValid: boolean;
    meta?: {
        modelInputSize?: [number, number];
        labelCounts?: { master: number; sub: number; article: number };
        timingsMs?: { preprocess: number; predict: number; total: number };
        warnings?: Record<string, string>;
    };
}
const prettyPct = (x?: number) =>
    typeof x === "number" ? `${(x * 100).toFixed(1)}%` : "–";

const Card: React.FC<{
    title: string;
    bucket: Bucket;
    showRestricted?: boolean;
}> = ({ title, bucket, showRestricted = true }) => (
    <div className="bg-white p-4 rounded-xl shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
        <div className="mt-1 text-gray-900">
            <span className="font-medium">Predicted:</span> {bucket.predicted}{" "}
            <span className="text-gray-600">
                ({prettyPct(bucket.confidence)})
            </span>
        </div>

        <div className="mt-3">
            <div className="text-sm font-medium text-gray-700">Top 3</div>
            <ul className="text-sm text-gray-700 list-disc ml-5">
                {bucket.top3?.map((t, i) => (
                    <li
                        key={i}
                    >{`${t.label} — ${prettyPct(t.confidence)} Confidence`}</li>
                ))}
            </ul>
        </div>

        {showRestricted && bucket.top3Restricted && (
            <div className="mt-3">
                <div className="text-sm font-medium text-gray-700">
                    Top 3 (restricted)
                </div>
                <ul className="text-sm text-gray-700 list-disc ml-5">
                    {bucket.top3Restricted.length ? (
                        bucket.top3Restricted.map((t, i) => (
                            <li key={i}>
                                {`${t.label} — ${prettyPct(t.confidence)} Confidence`}
                            </li>
                        ))
                    ) : (
                        <li>—</li>
                    )}
                </ul>
            </div>
        )}
    </div>
);

const ImageClassifierPage: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [result, setResult] = useState<ClassificationResult | null>(null);
    const [loading, setLoading] = useState(false);

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResult(null);
        }
    };

    const submitImage = async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append("image", selectedFile);

        setLoading(true);
        setResult(null);

        try {
            const response = await fetch("http://localhost:8000/api/classify", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(
                    `Failed to classify image (${response.status})`
                );
            }

            const data: ClassificationResult | { error: string } =
                await response.json();
            console.log("Classification response:", data);

            if ("error" in data) {
                alert(`Server error: ${data.error}`);
                return;
            }

            setResult(data);
        } catch (error) {
            console.error("Classification error:", error);
            alert("Error classifying image");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 flex items-center justify-center p-6">
            <div className="max-w-7xl w-full bg-white rounded-2xl shadow-lg p-8">
                <h1 className="text-3xl font-bold text-center text-gray-800 mb-6">
                    Fashion Image Classifier
                </h1>

                {/* Uploader */}
                <label className="flex flex-col items-center justify-center w-full h-64 border-4 border-dashed border-gray-300 rounded-xl cursor-pointer hover:border-blue-400 transition-colors">
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="hidden"
                    />
                    {previewUrl ? (
                        <img
                            src={previewUrl}
                            alt="preview"
                            className="w-full h-full object-contain rounded-xl"
                        />
                    ) : (
                        <div className="text-center text-gray-400">
                            <p className="text-lg font-medium">
                                Click or Drag & Drop an image
                            </p>
                            <p className="text-sm mt-2">
                                Supported formats: JPG, PNG, etc.
                            </p>
                        </div>
                    )}
                </label>

                {/* Button */}
                <button
                    onClick={submitImage}
                    className="mt-6 w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 rounded-xl shadow-md transition-colors disabled:opacity-50"
                    disabled={loading || !selectedFile}
                >
                    {loading ? "Classifying..." : "Classify Image"}
                </button>

                {/* Results */}
                {result && (
                    <div className="mt-8 space-y-6">
                        {/* Hierarchy status */}
                        <div
                            className={`p-3 rounded-xl ${
                                result.hierarchyValid
                                    ? "bg-green-50 text-green-700 border border-green-200"
                                    : "bg-red-50 text-red-700 border border-red-200"
                            }`}
                        >
                            {result.hierarchyValid
                                ? "Hierarchy is valid ✅ (master → sub → article)"
                                : "Hierarchy mismatch ⚠️ (predictions don’t align perfectly with training hierarchy)"}
                        </div>

                        {/* 3 columns */}
                        <div className="grid md:grid-cols-3 gap-4">
                            <Card
                                title="Master Category"
                                bucket={result.master}
                                showRestricted={false}
                            />
                            <Card title="Subcategory" bucket={result.sub} />
                            <Card
                                title="Article Type"
                                bucket={result.article}
                            />
                        </div>

                        {/* Meta (optional) */}
                        {/* {result.meta && (
                            <div className="bg-gray-50 p-4 rounded-xl border text-sm text-gray-700">
                                <div className="font-semibold mb-2">Meta</div>
                                <div className="grid md:grid-cols-3 gap-3">
                                    <div>
                                        <div className="font-medium">
                                            Model Input
                                        </div>
                                        <div>
                                            {result.meta.modelInputSize?.join(
                                                " x "
                                            )}{" "}
                                            (H x W)
                                        </div>
                                    </div>
                                    <div>
                                        <div className="font-medium">
                                            Label Counts
                                        </div>
                                        <div>
                                            master:{" "}
                                            {result.meta.labelCounts?.master} •
                                            sub: {result.meta.labelCounts?.sub}{" "}
                                            • article:{" "}
                                            {result.meta.labelCounts?.article}
                                        </div>
                                    </div>
                                    <div>
                                        <div className="font-medium">
                                            Timings
                                        </div>
                                        <div>
                                            preprocess:{" "}
                                            {result.meta.timingsMs?.preprocess}{" "}
                                            ms • predict:{" "}
                                            {result.meta.timingsMs?.predict} ms
                                            • total:{" "}
                                            {result.meta.timingsMs?.total} ms
                                        </div>
                                    </div>
                                </div>

                                {result.meta.warnings &&
                                    Object.keys(result.meta.warnings).length >
                                        0 && (
                                        <div className="mt-3">
                                            <div className="font-medium">
                                                Sanity Warnings
                                            </div>
                                            <ul className="list-disc ml-5">
                                                {Object.entries(
                                                    result.meta.warnings
                                                ).map(([k, v]) => (
                                                    <li key={k}>
                                                        <span className="font-medium">
                                                            {k}:
                                                        </span>{" "}
                                                        {v}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                            </div>
                        )} */}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageClassifierPage;
