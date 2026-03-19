using BepInEx;
using BepInEx.Logging;
using HarmonyLib;
using HutongGames.PlayMaker;
using System;
using System.Collections;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using UnityEngine;

[BepInPlugin("com.victorchristofoletti.dataextractor", "Hollow Knight Data Extractor", "1.0.0")]
public class DataExtractorMod : BaseUnityPlugin
{
    PlayerState playerState;

    private readonly object stateLock = new object();

    private bool isPlayerInScene = false;

    // Controle da nossa Thread em Background
    private Thread pipeThread;
    private bool isRunning = false;

    public static DataExtractorMod Instance;

    private AutoResetEvent frameReadyEvent = new AutoResetEvent(false);

    private void Awake()
    {
        Instance = this;
        var harmony = new Harmony("com.victorchristofoletti.dataextractor");
        harmony.PatchAll();
    }

    void Start()
    {
        // Initialize state object once to avoid Garbage Collection
        playerState = new PlayerState();

        isRunning = true;
        pipeThread = new Thread(PipeThreadLoop);

        pipeThread.IsBackground = true; // background process
        pipeThread.Start(); // initiating thread process here     
    }

    private void FixedUpdate()
    {
        isPlayerInScene = (GameManager.instance != null && GameManager.instance.IsGameplayScene());

        // Safeguard: Ensure player is actually in the game
        if (HeroController.instance == null || PlayerData.instance == null || !isPlayerInScene)
            return;

        lock (stateLock)
        {
            ExtractPlayerData();
        }

        frameReadyEvent.Set();
    }

    public void StartDeathReload(HeroController hero)
    {
        StartCoroutine(WaitAndReloadRoutine(hero));
    }

    private IEnumerator WaitAndReloadRoutine(HeroController hero)
    {
        yield return new WaitForSeconds(0.1f);

        PlayerData.instance.health = PlayerData.instance.maxHealth;
        PlayerData.instance.healthBlue = 0;

        string currentScene = GameManager.instance.sceneName;
        string entryGate = GameManager.instance.entryGateName;

        if (string.IsNullOrEmpty(entryGate)) entryGate = "door1";

        GameManager.instance.BeginSceneTransition(new GameManager.SceneLoadInfo
        {
            SceneName = currentScene,
            EntryGateName = GameManager.instance.entryGateName,
            EntryDelay = 0f,
            Visualization = GameManager.SceneLoadVisualizations.GodsAndGlory,
            PreventCameraFadeOut = true,
            AlwaysUnloadUnusedAssets = true
        });
    }

    // Thread Process...
    private void PipeThreadLoop()
    {
        while (isRunning)
        {
            try
            {
                // creating a name pipe called HK_RL_Pipe
                using (var pipeServer = new NamedPipeServerStream("HK_RL_Pipe", PipeDirection.Out))
                {
                    // waiting to be connected
                    pipeServer.WaitForConnection();

                    // creating a binary byte writer for the named pipe
                    using (var writer = new BinaryWriter(pipeServer))
                    {
                        while (isRunning && pipeServer.IsConnected) // if the process is running and // if the pipe is connected
                        {
                            if (isPlayerInScene) // if the player is in a scene
                            {
                                lock (stateLock)
                                {
                                    // writing and sending through the pipe

                                    // Position flags
                                    writer.Write(playerState.px); // player x pos
                                    writer.Write(playerState.py); // player y pos
                                    writer.Write(playerState.pvx); // player x velocity
                                    writer.Write(playerState.pvy); // player y velocity

                                    // Resources flags
                                    writer.Write(playerState.hp);
                                    writer.Write(playerState.maxHp);

                                    writer.Write(playerState.soul);
                                    writer.Write(playerState.maxSoul);
                                    //writer.Write(playerState.dead);

                                    // Facing flags
                                    writer.Write(playerState.facingRight);

                                    // Movement flags
                                    writer.Write(playerState.onGround);
                                    writer.Write(playerState.falling);
                                    writer.Write(playerState.wallJumping);
                                    writer.Write(playerState.jumping);
                                    writer.Write(playerState.doubleJumping);

                                    writer.Write(playerState.dashing);
                                    writer.Write(playerState.dashCooldown);

                                    writer.Write(playerState.shadowDashing);
                                    writer.Write(playerState.shadowDashCoolDown);

                                    // Behavior flags
                                    writer.Write(playerState.canFocus);
                                    writer.Write(playerState.focusing);
                                    writer.Write(playerState.invulnerable);

                                    // Attack flags
                                    writer.Write(playerState.isAttacking);
                                    writer.Write(playerState.attackDuration);
                                    writer.Write(playerState.attackRecoveryTime);
                                    writer.Write(playerState.attackCooldownTime);
                                    writer.Write(playerState.attackForward);
                                    writer.Write(playerState.attackUp);
                                    writer.Write(playerState.attackDown);
                                    writer.Write(playerState.canCast);
                                    writer.Write(playerState.casting);

                                    // Boss Data:

                                    // Positions flags
                                    writer.Write(playerState.bx);
                                    writer.Write(playerState.by);
                                    writer.Write(playerState.bvx);
                                    writer.Write(playerState.bvy);

                                    // Resources flags
                                    writer.Write(playerState.bossHp);
                                    writer.Write(playerState.bossMaxHp);

                                    // Current boss flag
                                    //writer.Write(playerState.bossSceneHash);

                                    // Behavior flags
                                    //writer.Write(playerState.bossIsDead);
                                    writer.Write(playerState.bossState);
                                    writer.Write(playerState.bossStateAmount);
                                    writer.Write(playerState.bossSceneHash);
                                }

                                writer.Flush();
                            }

                            //Thread.Sleep(16);
                            frameReadyEvent.WaitOne();
                        }
                    }
                }
            }
            catch (Exception)
            {
                Thread.Sleep(1000);
            }
        }
    }

    private void ExtractPlayerData()
    {
        // Position 
        playerState.px = HeroController.instance.transform.position.x;
        playerState.py = HeroController.instance.transform.position.y;
        playerState.pvx = HeroController.instance.current_velocity.x;
        playerState.pvy = HeroController.instance.current_velocity.y;

        // Resources
        playerState.hp = PlayerData.instance.health;
        playerState.maxHp = PlayerData.instance.maxHealth;
        playerState.soul = PlayerData.instance.MPCharge;
        playerState.maxSoul = PlayerData.instance.maxMP;
        //playerState.dead = HeroController.instance.cState.dead;

        // Facing flags
        playerState.facingRight = HeroController.instance.cState.facingRight;

        // Machine States
        playerState.onGround = HeroController.instance.cState.onGround;
        playerState.falling = HeroController.instance.cState.falling;
        playerState.wallJumping = HeroController.instance.cState.wallJumping;
        playerState.jumping = HeroController.instance.cState.jumping;
        playerState.doubleJumping = HeroController.instance.cState.doubleJumping;


        playerState.dashing = HeroController.instance.cState.dashing;
        playerState.dashCooldown = HeroController.instance.DASH_COOLDOWN;

        playerState.shadowDashing = HeroController.instance.cState.shadowDashing;
        playerState.shadowDashCoolDown = HeroController.instance.SHADOW_DASH_COOLDOWN;

        // Behavior flags
        playerState.canFocus = HeroController.instance.CanFocus();
        playerState.focusing = HeroController.instance.cState.focusing;
        playerState.invulnerable = HeroController.instance.cState.invulnerable;

        // Attack States
        playerState.isAttacking = HeroController.instance.cState.attacking;
        playerState.attackDuration = HeroController.instance.ATTACK_DURATION;
        playerState.attackRecoveryTime = HeroController.instance.ATTACK_RECOVERY_TIME;
        playerState.attackCooldownTime = HeroController.instance.ATTACK_COOLDOWN_TIME;

        if (!HeroController.instance.cState.upAttacking &&
            !HeroController.instance.cState.downAttacking
        )
        {
            playerState.attackForward = HeroController.instance.cState.attacking;
        }

        playerState.attackUp = HeroController.instance.cState.upAttacking;
        playerState.attackDown = HeroController.instance.cState.downAttacking;


        playerState.canCast = HeroController.instance.CanCast();
        playerState.casting = HeroController.instance.cState.casting;

        // Boss Data:
        ExtractBossData();
    }

    private void ExtractBossData()
    {
        if (GameManager.instance != null)
            playerState.bossSceneHash = GameManager.instance.sceneName.GetHashCode();

        if (BossSceneController.IsBossScene && BossSceneController.Instance.bosses != null)
        {
            for (int i = 0; i < BossSceneController.Instance.bosses.Length; i++)
            {
                HealthManager boss = BossSceneController.Instance.bosses[i];

                if (boss != null)
                {
                    PlayMakerFSM[] fsms = boss.gameObject.GetComponents<PlayMakerFSM>();
                    var bossRb = boss.GetComponent<Rigidbody2D>();

                    foreach (PlayMakerFSM fsm in fsms)
                    {
                        if (fsm.FsmName == "Control")
                        {
                            // 1. Pega o nome do estado atual
                            string stateName = fsm.ActiveStateName;

                            // 2. Encontra o 'int' (índice) desse estado dentro do array do PlayMaker
                            int stateInt = -1;
                            for (int j = 0; j < fsm.FsmStates.Length; j++)
                            {
                                if (fsm.FsmStates[j].Name == stateName)
                                {
                                    stateInt = j; // Encontrou o seu int de referência!
                                    break;
                                }
                            }

                            Time.timeScale = 3f;

                            // Position flags
                            playerState.bx = boss.transform.position.x;
                            playerState.by = boss.transform.position.y;

                            playerState.bvx = bossRb ? bossRb.linearVelocityX : 0;
                            playerState.bvy = bossRb ? bossRb.linearVelocityY : 0;

                            // Resource flags
                            playerState.bossHp = boss.hp;

                            if (playerState.bossHp > playerState.bossMaxHp)
                                playerState.bossMaxHp = playerState.bossHp;

                            //playerState.bossIsDead = boss.isDead;

                            // Behavior flags
                            playerState.bossState = stateInt;
                            playerState.bossStateAmount = fsm.FsmStates.Length;
                        }
                    }
                }
            }
        }
    }

    void OnDestroy()
    {
        // Clean up the port when the game closes or mod is disabled
        isRunning = false;
        frameReadyEvent.Set();
    }
}

[HarmonyPatch(typeof(HeroController), "Die")]
public class HeroControllerDiePatch
{
    [HarmonyPrefix]
    public static bool Prefix(HeroController __instance)
    {
        DataExtractorMod.Instance?.StartDeathReload(__instance);

        return false;
    }
}

