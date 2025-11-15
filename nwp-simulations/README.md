This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## To do
- refactor in the steps like advect_scalar, diffuse_velocity the repeated create bind group calls. io type handles shuffling between s0 for rk2 step 1 and s_star for rk2 step 2. if you create all theose buffers from step rk2 in a central place and then create buffers for like rk2_step1_thisTypeBuffer and rk2_step2_thisTypeBuffer then you can avoid the memory problems by dozens of create bind groups scattered in dispatch functions. would maybe help firefox stay stable like chrome does

- making the render 3d
	- can start with slider that adjusts which slice is being shown
- finish step rk2 function
- make camera to move around
- render clouds and stuff
- i guess the most important view isn't temperature or moisture but simply cloud view on a sky