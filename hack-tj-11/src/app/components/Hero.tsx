'use client';

import Image from "next/image";
import CLOUDS from 'vanta/dist/vanta.clouds.min';
import React, { ReactNode, useEffect, useRef, useState } from 'react'

export default function Hero({ children: ReactNode }) {
  const [vantaEffect, setVantaEffect] = useState(null)
  const myRef = useRef(null)
  useEffect(() => {
    if (!vantaEffect) {
      setVantaEffect(CLOUDS({
        el: myRef.current
      }))
    }
    return () => {
      if (vantaEffect) vantaEffect.destroy()
    }
  }, [vantaEffect])
  return (
    <>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/vanta/dist/vanta.waves.min.js"></script>
      <main ref={myRef} className="bg-blend-darken bg-black flex min-h-screen flex-col items-center justify-center p-24">
        {children}
      </main>
    </>
  );
}
