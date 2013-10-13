        program hc_widefield
        
c
c       Fortran source code, developed with an Intel Fortran90 compiler on Linux
c
c       note: implicit type convention: variables starting with i..n are
c       integers, all others are reals
c
c       9 Jan 2005, author J.H. van Hateren
c       Belongs to "A cellular and molecular model of response kinetics
c       and adaptation in primate cones and horizontal cells", J.Vision (2005)
c
c       Remarks:
c        -function rtbis not yet included (see below); insert that,
c         and change the statements for resp_c and resp_vs at 
c         'set pre-stimulus values'; the values there now are only 
c         correct for stim_prev=100 td
c       - for higher gains in the feedback loops, nrate may need to 
c         be larger
c       - check the ARMA coefficients for very high or very small values
c         of tau, switch to double precision variables if necessary
c
                  
        parameter (nrate=10000)                !#timesteps/sec (delta_t=100 us)

        parameter (nstim_max=5*nrate)          !maximum stimulus length 5 sec

        real stim(2)                           !I in model
        real resp_r(2)                         !signal after tau_r
        real resp_e(2)                         !E* in model
        real beta(2)                           !beta in model
        real resp_q(2)                         !Q in model
        real tau_x(2)                          !tau_x in model
        real resp_w(2)                         !alpha/beta in model
        real gain_x(2)                         !alpha in model
        real resp_x(2)                         !X in model
        real resp_os(2)                        !I_os in model
        real resp_c(2)                         !C in model
        real resp_vc(2)                        !signal after division by g_i
        real resp_is(2)                        !V_is in model
        real gain_is(2)                        !g_is in model
        real gain_i(2)                         !g_i in model
        real resp_vs(2)                        !V_s in model
        real resp_ic(2)                        !I_t in model
        real resp_tr(2)                        !signal after tau_1
        real resp_ih(2)                        !signal after tau_2 (V_b)
        real resp_h(2)                         !V_h in model
        real resp_h0(2)                        !V_is_prime in model
        real stim_ar(nstim_max)                !stimulus array
        real resp_ar(nstim_max)                !response array        
        real resp_kl(nstim_max)                !auxiliary variable used for delay      
        
        common /csteady/ a_c,rnc,beta_0,g_ex0,stim_prev,rnx,st_is,
     &gripmax,vk,vn                   !global variables needed by steady/steady_vs
        
        external steady,steady_vs              !needed when rtbis is used

        ratefrac=float(nrate)/1000.            !conversion 1 ms to 100 us timebase

c
c       make stimulus
c                
        nlen=ifix(300.*ratefrac)            !300 ms stimulus duration
        ip1=ifix(25.*ratefrac)              !25 ms stimulus latency
        ip2=ip1+ifix(100.*ratefrac)-1       !100 ms pulse width
        stim_dc=100.                        !background illuminance 100 td
        stim_prev=100.                      !assumed illuminance before start stimulus
        stim_ar(1:nlen)=stim_dc
        stim_ar(ip1:ip2)=stim_dc+2.*stim_dc     !contrast 2

c
c       parameters, in ms where applicable, converted to delta_t by ratefrac;
c       values are those of Figs.7 and 6A of the article
c
        tau_r=0.49*ratefrac                 !parameter tau_r in model
        tau_e=16.8*ratefrac                 !parameter tau_e in model
        beta_0=2.80e-3                      !parameter c_beta in model
        g_ex0=1.63e-4                       !parameter k_beta in model
        rnx=1.                              !parameter n_x in model
        rnc=4.                              !parameter n_c in model
        tau_c=2.89*ratefrac                 !parameter tau_c in model
        a_c=9.08e-2                         !parameter a_c in model
        tau_vc=4.*ratefrac                  !parameter tau_m in model
        gamma_is=0.678                      !parameter gamma_is in model
        tau_is=56.9*ratefrac                !parameter tau_is in model
        a_is=7.09e-2                        !parameter a_is in model
        ripmax=151.1                        !parameter g_t in model
        vk=-10.                             !parameter v_k in model
        vn=3.                               !parameter v_n in model
        tau_tr=4.*ratefrac                  !parameter tau_1 in model 
        tau_ih=4.*ratefrac                  !parameter tau_2 in model 
        tau_h=20.*ratefrac                  !parameter tau_h in model 
        rdel=2.82*ratefrac                  !overall delay
        vh0=19.7                            !parameter V_I in model 
        rho=0.733                           !parameter mu in model 
        tau_h0=250.*ratefrac                !parameter tau_a in model 

c
c       calculate ARMA coefficients of fixed filters
c
        f1_tau_r=exp(-1./tau_r)
        f2_tau_r=(tau_r-(1.+tau_r)*f1_tau_r)
        f3_tau_r=(1.-tau_r+tau_r*f1_tau_r)

        f1_tau_e=exp(-1./tau_e)
        f2_tau_e=(tau_e-(1.+tau_e)*f1_tau_e)
        f3_tau_e=(1.-tau_e+tau_e*f1_tau_e)

        f1_tau_c=exp(-1./tau_c)
        f2_tau_c=(tau_c-(1.+tau_c)*f1_tau_c)
        f3_tau_c=(1.-tau_c+tau_c*f1_tau_c)

        f1_tau_vc=exp(-1./tau_vc)
        f2_tau_vc=(tau_vc-(1.+tau_vc)*f1_tau_vc)
        f3_tau_vc=(1.-tau_vc+tau_vc*f1_tau_vc)

        f1_tau_is=exp(-1./tau_is)
        f2_tau_is=a_is*(tau_is-(1.+tau_is)*f1_tau_is)
        f3_tau_is=a_is*(1.-tau_is+tau_is*f1_tau_is)

        f1_tau_tr=exp(-1./tau_tr)
        f2_tau_tr=tau_tr-(1.+tau_tr)*f1_tau_tr
        f3_tau_tr=1.-tau_tr+tau_tr*f1_tau_tr

        f1_tau_h0=exp(-1./tau_h0)
        f2_tau_h0=tau_h0-(1.+tau_h0)*f1_tau_h0
        f3_tau_h0=1.-tau_h0+tau_h0*f1_tau_h0

        do it=1,nlen                       !main loop

c
c       ncurr and nprev determine which element of, e.g., resp_r, is
c       the current one, resp_r(1) or resp_r(2)
c
         if (it.eq.1) then
          ncurr=1                          !current
          nprev=2                          !previous
         else
          nkl=ncurr                        !swap values ncurr and nprev
          ncurr=nprev
          nprev=nkl
         endif

         if (it.eq.1) then                 !set pre-stimulus values
c          resp_c(nprev)=rtbis(steady,0.,1.e3,1.e-9)        !find steady-state C
          resp_c(nprev)=14.12579           !delete if rtbis is available
          stim(nprev)=stim_prev
          resp_r(nprev)=stim(nprev)
          resp_e(nprev)=resp_r(nprev)
          beta(nprev)=beta_0+g_ex0*resp_e(nprev)
          resp_q(nprev)=1./beta(nprev)
          tau_x(nprev)=resp_q(nprev)*ratefrac
          gain_x(nprev)=1./(1.+(a_c*resp_c(nprev))**rnc)
          resp_w(nprev)=gain_x(nprev)*resp_q(nprev)
          resp_x(nprev)=gain_x(nprev)*resp_q(nprev)
          resp_os(nprev)=resp_x(nprev)**rnx
          resp_vc(nprev)=(resp_os(nprev)/a_is)**(1./(1.+gamma_is))                
          resp_is(nprev)=resp_vc(nprev)
          gain_is(nprev)=resp_is(nprev)**gamma_is
          gain_i(nprev)=a_is*gain_is(nprev)
          resp_h0(nprev)=resp_is(nprev)
          gtau=(resp_h0(nprev)/vh0)**rho                   !a_I in model
          gripmax=ripmax/gtau
          st_is=resp_is(nprev)                !transferred to steady_vs via common
c          resp_vs(nprev)=rtbis(steady_vs,-1.e3,1.e3,1.e-9)   !find steady-state V_s
          resp_vs(nprev)=-12.92790            !delete if rtbis is available
          resp_ic(nprev)=gripmax/(1.+exp(-(resp_vs(nprev)-vk)/vn))
          resp_tr(nprev)=resp_ic(nprev)
          resp_ih(nprev)=resp_tr(nprev)
          resp_h(nprev)=resp_ih(nprev)
         endif

         stim(ncurr)=stim_ar(it)

         resp_r(ncurr)=f1_tau_r*resp_r(nprev)+
     &f2_tau_r*stim(nprev)+
     &f3_tau_r*stim(ncurr)

         resp_e(ncurr)=f1_tau_e*resp_e(nprev)+
     &f2_tau_e*resp_r(nprev)+
     &f3_tau_e*resp_r(ncurr)

         beta(ncurr)=beta_0+g_ex0*resp_e(ncurr)
         resp_q(ncurr)=1./beta(ncurr)

         tau_x(ncurr)=resp_q(ncurr)*ratefrac
         f1_tau_x=exp(-1./tau_x(ncurr))                   !ARMA coefficients tau_x
         f2_tau_x=tau_x(ncurr)-(1.+tau_x(ncurr))*f1_tau_x
         f3_tau_x=1.-tau_x(ncurr)+tau_x(ncurr)*f1_tau_x

         resp_x(ncurr)=f1_tau_x*resp_x(nprev)+
     &gain_x(nprev)*f2_tau_x*resp_q(nprev)+
     &gain_x(nprev)*f3_tau_x*resp_q(ncurr)

         resp_w(ncurr)=gain_x(nprev)*resp_q(ncurr)        !not necessary, only for
                                                          !figure

         resp_os(ncurr)=resp_x(ncurr)**rnx

         resp_c(ncurr)=f1_tau_c*resp_c(nprev)+
     &f2_tau_c*resp_os(nprev)+
     &f3_tau_c*resp_os(ncurr)

         gain_x(ncurr)=1./(1.+(a_c*resp_c(ncurr))**rnc)

         resp_vc(ncurr)=resp_os(ncurr)/gain_i(nprev) 

         resp_is(ncurr)=f1_tau_vc*resp_is(nprev)+
     &f2_tau_vc*resp_vc(nprev)+
     &f3_tau_vc*resp_vc(ncurr)

         gain_is(ncurr)=resp_is(ncurr)**gamma_is

         gain_i(ncurr)=f1_tau_is*gain_i(nprev)+
     &f2_tau_is*gain_is(nprev)+f3_tau_is*gain_is(ncurr)      

         resp_h0(ncurr)=f1_tau_h0*resp_h0(nprev)+
     &f2_tau_h0*resp_is(nprev)+
     &f3_tau_h0*resp_is(ncurr)

         gtau=(resp_h0(ncurr)/vh0)**rho

         gripmax=ripmax/gtau
         gtau_ih=tau_ih*gtau                            !tau_2_prime in model
         gtau_h=tau_h*gtau                              !tau_h_prime in model

         resp_vs(ncurr)=resp_is(ncurr)-resp_h(nprev)

         resp_ic(ncurr)=gripmax/(1.+exp(-(resp_vs(ncurr)-vk)/vn))

         resp_tr(ncurr)=f1_tau_tr*resp_tr(nprev)+
     &f2_tau_tr*resp_ic(nprev)+
     &f3_tau_tr*resp_ic(ncurr)

         f1_tau_ih=exp(-1./gtau_ih)                !ARMA coefficients tau_2_prime
         f2_tau_ih=gtau_ih-(1.+gtau_ih)*f1_tau_ih
         f3_tau_ih=1.-gtau_ih+gtau_ih*f1_tau_ih

         f1_tau_h=exp(-1./gtau_h)
         f2_tau_h=gtau_h-(1.+gtau_h)*f1_tau_h
         f3_tau_h=1.-gtau_h+gtau_h*f1_tau_h

         resp_ih(ncurr)=f1_tau_ih*resp_ih(nprev)+
     &f2_tau_ih*resp_tr(nprev)+
     &f3_tau_ih*resp_tr(ncurr)

         resp_h(ncurr)=f1_tau_h*resp_h(nprev)+
     &f2_tau_h*resp_ih(nprev)+
     &f3_tau_h*resp_ih(ncurr)

         resp_ar(it)=resp_h(ncurr)                !output of resp_h (replace
                                                  !for obtaining other signals)

        enddo                                     !end of main loop
 
c
c       delay response by interpolation
c
        do it=1,nlen
         rel=float(it)-rdel
         if (rel.ge.0.) then
          iel1=ifix(rel)
          iel2=iel1+1
          d1=rel-float(iel1)
          d2=1.-d1
         else
          iel1=ifix(rel)
          iel2=iel1-1
          d1=float(iel1)-rel
          d2=1.-d1
         endif
         if (iel1.lt.1) iel1=1
         if (iel1.gt.nlen) iel1=nlen
         if (iel2.lt.1) iel2=1
         if (iel2.gt.nlen) iel2=nlen
         resp_kl(it)=d2*resp_ar(iel1)+d1*resp_ar(iel2)
        enddo
        resp_ar(1:nlen)=resp_kl(1:nlen)
        
c
c       save stimulus and response
c
        open (11,file='stimulus',status='unknown')
        open (12,file='response',status='unknown')
        do it=1,nlen
         time=float(it-1)/ratefrac        !time in ms
         write (11,*) time,stim_ar(it)
         write (12,*) time,resp_ar(it)
        enddo
        close (11)
        close (12) 

        stop
        end 


        function steady(x)
        common /csteady/ a_c,rnc,beta_0,g_ex0,stim_prev,rnx,st_is,
     &gripmax,vk,vn
        steady=x-(1./(1.+(a_c*x)**rnc)/
     &(beta_0+g_ex0*stim_prev))**rnx
        return
        end


        function steady_vs(x)
        common /csteady/ a_c,rnc,beta_0,g_ex0,stim_prev,rnx,st_is,
     &gripmax,vk,vn
        steady_vs=x-(st_is-gripmax/(1.+exp(-(x-vk)/vn)))
        return
        end


      FUNCTION rtbis(func,x1,x2,xacc)
      INTEGER JMAX
      REAL rtbis,x1,x2,xacc,func
      EXTERNAL func
      PARAMETER (JMAX=40)
      INTEGER j
      REAL dx,f,fmid,xmid
      fmid=func(x2)
      f=func(x1)
      if(f*fmid.ge.0.) pause 'root must be bracketed in rtbis'
      if(f.lt.0.)then
        rtbis=x1
        dx=x2-x1
      else
        rtbis=x2
        dx=x1-x2
      endif
      do 11 j=1,JMAX
        dx=dx*.5
        xmid=rtbis+dx
        fmid=func(xmid)
        if(fmid.le.0.)rtbis=xmid
        if(abs(dx).lt.xacc .or. fmid.eq.0.) return
11    continue
      pause 'too many bisections in rtbis'
      END
C  (C) Copr. 1986-92 Numerical Recipes Software v%1jw#<0(9p#3.