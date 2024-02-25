'use client';

import Image from "next/image";
import CLOUDS from 'vanta/dist/vanta.clouds.min';
import React, { useEffect, useRef, useState } from 'react'

export default function HeroSection({ children }) {
  const [vantaEffect, setVantaEffect] = useState(null)
  const myRef = useRef(null)
  useEffect(() => {
    try {
      if (!vantaEffect && myRef && myRef.current) {
        setVantaEffect(CLOUDS({
          el: myRef.current
        }))
      }
    } catch (e) {
      console.log(e)
    }
    return () => {
      if (vantaEffect) vantaEffect.destroy()
    }
  }, [myRef, vantaEffect])
  return (
    <>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/vanta/dist/vanta.waves.min.js"></script>
      <main ref={myRef} className="absolute top-0 opacity-50 min-h-screen w-full">
        {/* {children} */}
      </main>
      <main className="flex min-h-screen flex-col items-center justify-center p-24">
        {children}
      </main>
    </>
  );
}
