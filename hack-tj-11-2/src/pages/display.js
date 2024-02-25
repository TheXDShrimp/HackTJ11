import Head from 'next/head';
import Navbar from "../components/Navbar";

export default function Home() {
    return (
        <>
            <Navbar />
            <div className="flex justify-center items-center h-screen">
                <Head>
                    <title>Quadrant Images</title>
                </Head>

                <div className="grid grid-cols-2 grid-rows-2 gap-4">
                    <div className="flex flex-col items-center justify-center">
                        <img src="/orig.jpg" alt="File 1" className="w-64 h-64" />
                        <span className="mt-2">Original Image</span>
                    </div>

                    <div className="flex flex-col items-center justify-center">
                        <img src="/adversarialLow.jpg" alt="File 2" className="w-64 h-64" />
                        <span className="mt-2">Adversarially Attacked</span>
                    </div>

                    <div className="flex flex-col items-center justify-center">
                        <img src="/origDeepfakeNOGOOD.jpg" alt="File 3" className="w-64 h-64" />
                        <span className="mt-2">Deepfaked Original</span>
                    </div>

                    <div className="flex flex-col items-center justify-center">
                        <img src="/adversarialDeepfakeLBOZO.jpg" alt="File 4" className="w-64 h-64" />
                        <span className="mt-2">Attempted Deepfake</span>
                    </div>
                </div>
            </div>
        </>
    );
}
